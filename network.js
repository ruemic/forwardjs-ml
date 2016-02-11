'use strict';

var math = require('forwardjs-ml-math');
var Layer = require('./layer');

module.exports = class Network {
  constructor(/* layer sizes */) {
    var sizes = [...arguments];
    this.layers = [];
    for (var i = 1; i < arguments.length; i++) {
      var layer = new Layer(sizes[i], sizes[i-1]+1)
      this.layers.push(layer);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Array}
   */
  forward(inputs) {
    var nextLayersInputs = inputs;
    for (var i = 0; i < this.layers.length; i++) {
      var layer = this.layers[i];
      nextLayersInputs = layer.forward(
        [1].concat(nextLayersInputs)
      );
    }
    return nextLayersInputs;
  }

  /**
   * @param {Array} errors
   */
  backward(errors) {
    for (var i = this.layers.length - 1; i >= 0; i--) {
      var layer = this.layers[i];
      errors = layer.backward(errors);
    }
  }

  updateWeights() {
    this.layers.forEach(l => l.updateWeights());
  }
}
