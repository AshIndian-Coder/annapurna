import 'package:image/image.dart' as img;
import 'dart:typed_data';

class QualityGate {
  static const double blurThreshold = 100.0;
  static const double minBrightness = 40.0;
  static const double maxBrightness = 240.0;

  static String? checkImageQuality(Uint8List imageBytes) {
    final image = img.decodeImage(imageBytes);
    if (image == null) return 'Invalid image format';

    double totalBrightness = 0;
    for (final pixel in image) {
      totalBrightness += _getLuminance(pixel);
    }
    final avgBrightness = totalBrightness / (image.width * image.height);

    if (avgBrightness < minBrightness) return 'Retake: too dark';
    if (avgBrightness > maxBrightness) return 'Retake: too bright';

    final variance = _calculateLaplacianVariance(image);
    if (variance < blurThreshold) return 'Retake: blurry';

    return null;
  }

  static double _getLuminance(img.Pixel pixel) {
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
  }

  static double _calculateLaplacianVariance(img.Image image) {
    final grayscale = img.grayscale(image);
    double sum = 0;
    double sqSum = 0;
    int count = 0;

    for (int y = 1; y < grayscale.height - 1; y++) {
      for (int x = 1; x < grayscale.width - 1; x++) {
        final center = _getLuminance(grayscale.getPixel(x, y));
        final top = _getLuminance(grayscale.getPixel(x, y - 1));
        final bottom = _getLuminance(grayscale.getPixel(x, y + 1));
        final left = _getLuminance(grayscale.getPixel(x - 1, y));
        final right = _getLuminance(grayscale.getPixel(x + 1, y));

        final laplacian = (top + bottom + left + right - 4 * center).abs();
        sum += laplacian;
        sqSum += laplacian * laplacian;
        count++;
      }
    }

    if (count == 0) return 0.0;
    final mean = sum / count;
    return (sqSum / count) - (mean * mean);
  }
}
