import 'dart:typed_data';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final tfliteProvider = Provider<TfliteRunner>((ref) => TfliteRunner());

class TfliteRunner {
  Future<void> loadModel() async {
    // Mock loading
  }

  Future<String> runInference(Float32List inputTensor) async {
    return 'Simulation: Looks OK';
  }

  void dispose() {}
}
