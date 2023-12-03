
import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
model.fit(tf.tensor2d([1, 2, 3, 4], [4, 1]), tf.tensor2d([1, 3, 5, 7], [4, 1]), {epochs: 10}).then(() => {
  const result = model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor;
  result.print();
});


model.save('file://./model').then(() => {
  console.log('model saved')
})
