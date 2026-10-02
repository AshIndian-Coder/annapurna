import 'dart:typed_data';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:image/image.dart' as img;

class ImagePreprocessor {
  static Future<Uint8List?> compressAndStripExif(String imagePath) async {
    return await FlutterImageCompress.compressWithFile(
      imagePath,
      minWidth: 1600,
      minHeight: 1600,
      quality: 80,
      keepExif: false,
    );
  }

  static Float32List imageToTensor(Uint8List imageBytes) {
    final image = img.decodeImage(imageBytes);
    if (image == null) throw Exception('Failed to decode image');

    final resized = img.copyResize(image, width: 224, height: 224);
    final tensor = Float32List(224 * 224 * 3);

    int idx = 0;
    for (final pixel in resized) {
      tensor[idx++] = pixel.r / 255.0;
      tensor[idx++] = pixel.g / 255.0;
      tensor[idx++] = pixel.b / 255.0;
    }
    
    return tensor;
  }
}
