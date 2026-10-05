import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';
import 'litert.dart';
import 'preprocess.dart';
import 'quality_gate.dart';

class CapturePreviewScreen extends ConsumerStatefulWidget {
  final String batchId;
  const CapturePreviewScreen({super.key, required this.batchId});

  @override
  ConsumerState<CapturePreviewScreen> createState() => _CapturePreviewScreenState();
}

class _CapturePreviewScreenState extends ConsumerState<CapturePreviewScreen> {
  CameraController? _controller;
  bool _isProcessing = false;
  String? _gateError;
  String? _inferenceResult;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;

      _controller = CameraController(
        cameras.first,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Future<void> _capture() async {
    if (_controller == null || !_controller!.value.isInitialized || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      _gateError = null;
      _inferenceResult = null;
    });

    try {
      final file = await _controller!.takePicture();
      final compressedBytes = await ImagePreprocessor.compressAndStripExif(file.path);
      
      if (compressedBytes == null) throw Exception();

      final gateResult = QualityGate.checkImageQuality(compressedBytes);
      if (gateResult != null) {
        setState(() => _gateError = gateResult);
        return;
      }

      final tensor = ImagePreprocessor.imageToTensor(compressedBytes);
      final tflite = ref.read(tfliteProvider);
      final result = await tflite.runInference(tensor);

      setState(() => _inferenceResult = result);
      
    } catch (e) {
      setState(() => _gateError = 'Capture failed. Try again.');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(backgroundColor: Colors.black),
        body: const Center(child: Text('Camera initializing...', style: TextStyle(color: Colors.white))),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: 3 / 4,
              child: CameraPreview(_controller!),
            ),
          ),
          
          if (_gateError != null)
            Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(
                  color: AppColors.danger.withOpacity(0.9),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _gateError!,
                  style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ),
            ),

          if (_inferenceResult != null)
            Positioned(
              top: 60,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surface.withOpacity(0.9),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    Text(
                      _inferenceResult!,
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        color: _inferenceResult == 'Looks OK' || _inferenceResult!.contains('Simulation') ? AppColors.good : AppColors.warning,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text('Preview — final decision after upload', style: TextStyle(color: AppColors.textSecondary, fontSize: 12)),
                    const SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        onPressed: () {
                          context.go('/kitchen/quality/result/${widget.batchId}');
                        },
                        child: const Text('Upload to Server'),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _capture,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4),
                    color: _isProcessing ? AppColors.textMuted : Colors.white.withOpacity(0.5),
                  ),
                  child: _isProcessing ? const Center(child: CircularProgressIndicator(color: Colors.white)) : null,
                ),
              ),
            ),
          ),
          
          Positioned(
            top: 40,
            left: 8,
            child: IconButton(
              icon: const Icon(Icons.close, color: Colors.white, size: 32),
              onPressed: () => context.pop(),
            ),
          ),
        ],
      ),
    );
  }
}
