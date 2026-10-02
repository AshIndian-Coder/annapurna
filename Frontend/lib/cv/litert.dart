import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final tfliteProvider = Provider<TfliteRunner>((ref) => TfliteRunner());

class TfliteRunner {
  Interpreter? _interpreter;

  Future<void> loadModel() async {
    if (_interpreter != null) return;
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/model_int8.tflite');
    } catch (_) {}
  }

  Future<String> runInference(Float32List inputTensor) async {
    if (_interpreter == null) await loadModel();
    if (_interpreter == null) return 'Simulation: Looks OK';
    
    final output = List<double>.filled(3, 0).reshape([1, 3]);
    final input = inputTensor.reshape([1, 224, 224, 3]);

    _interpreter!.run(input, output);

    final scores = output[0] as List<double>;
    
    int maxIdx = 0;
    double maxScore = scores[0];
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIdx = i;
      }
    }

    if (maxIdx == 0) return 'Looks OK';
    if (maxIdx == 1) return 'Possible risk';
    return 'Not sure';
  }

  void dispose() {
    _interpreter?.close();
  }
}
