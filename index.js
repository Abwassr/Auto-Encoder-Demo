console.log("Hello Autoencoder 🐯");

import * as tf from "@tensorflow/tfjs-node";
import { buffer, train } from "@tensorflow/tfjs-node";
// import canvas from "canvas";
// const { loadImage } = canvas;
import Jimp from "jimp";
import numeral from "numeral";

main();

async function main() {
  const autoencoder = buildModel();

  const images = await loadImages(550);

  const x_train = tf.tensor2d(images.slice(0, 500));

  await trainModel(autoencoder, x_train, 250);

  const x_test = tf.tensor2d(images.slice(500));
  await generateTests(autoencoder, x_test);
}
async function generateTests(autoencoder, x_test) {
  const output = autoencoder.predict(x_test);

  const newImages = await output.array();

  for (let i = 0; i < newImages.length; i++) {
    const img = newImages[i];
    const buffer = [];
    for (let n = 0; n < img.length; n++) {
      buffer[n * 4 + 0] = img[n] * 255;
      buffer[n * 4 + 1] = img[n] * 255;
      buffer[n * 4 + 2] = img[n] * 255;
      buffer[n * 4 + 3] = 255;
    }
    const image = await new Jimp({
      data: Buffer.from(buffer),
      width: 28,
      height: 28,
    });

    const num = numeral(i).format("000");
    image.write(`output/square${num}.png`);
  }
}

function buildModel() {
  const autoencoder = tf.sequential();

  autoencoder.add(
    tf.layers.dense({
      units: 256,
      inputShape: [784],
      activation: "relu",
    })
  );
  autoencoder.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    })
  );

  autoencoder.add(
    tf.layers.dense({
      units: 256,
      activation: "sigmoid",
    })
  );
  autoencoder.add(
    tf.layers.dense({
      units: 784,
      activation: "sigmoid",
    })
  );
  autoencoder.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return autoencoder;
}

async function loadImages(total) {
  const allImages = [];

  for (let i = 0; i < total; i++) {
    const num = numeral(i).format("000");
    const img = await Jimp.read(`data/square${num}.png`);
    const rawData = [];
    for (let n = 0; n < 28 * 28; n++) {
      let index = n * 4;
      let r = img.bitmap.data[index + 0];
      rawData[n] = r / 255.0;
    }
    allImages[i] = rawData;
  }
  return allImages;
}

async function trainModel(autoencoder, x_train, epochs) {
  await autoencoder.fit(x_train, x_train, {
    epochs: epochs,
    batchSize: 256,
    shuffle: true,
    verbose: true,
  });
}
