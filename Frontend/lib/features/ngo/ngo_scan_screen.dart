import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/ngo_service.dart';
import '../../shared/widgets/profile_drawer.dart';
import '../../shared/widgets/qr_scanner_screen.dart';

class NgoScanScreen extends ConsumerStatefulWidget {
  const NgoScanScreen({super.key});

  @override
  ConsumerState<NgoScanScreen> createState() => _NgoScanScreenState();
}

class _NgoScanScreenState extends ConsumerState<NgoScanScreen> {
  final TextEditingController _manualController = TextEditingController();
  bool _isLoading = false;

  Future<void> _processScan(String code) async {
    if (_isLoading || code.isEmpty) return;
    setState(() => _isLoading = true);

    final service = ref.read(ngoServiceProvider);
    final eventId = const Uuid().v7();
    
    final result = await service.receiveDelivery(code, clientEventId: eventId);
    
    if (!mounted) return;
    setState(() => _isLoading = false);

    result.when(
      success: (_) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Delivery marked as received successfully!'), backgroundColor: AppColors.good),
        );
        _manualController.clear();
      },
      failure: (error) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(error.message), backgroundColor: AppColors.danger),
        );
      },
    );
  }

  @override
  void dispose() {
    _manualController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Receive Delivery'),
      ),
      drawer: const ProfileDrawer(),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Icon(Icons.qr_code_scanner, size: 120, color: AppColors.textSecondary),
            const SizedBox(height: 24),
            Text(
              'Scan Driver QR',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            const Text(
              'Ask the driver to present their delivery QR code and scan it to confirm receipt.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 48),
            FilledButton.icon(
              onPressed: _isLoading ? null : () async {
                final String? code = await Navigator.push<String>(
                  context,
                  MaterialPageRoute(builder: (context) => const QrScannerScreen()),
                );
                if (code != null && code.isNotEmpty) {
                  _processScan(code);
                }
              },
              icon: const Icon(Icons.camera_alt),
              label: const Text('Open Camera Scanner'),
            ),
            const SizedBox(height: 32),
            const Divider(),
            const SizedBox(height: 32),
            TextField(
              controller: _manualController,
              decoration: const InputDecoration(
                labelText: 'Manual Batch Code',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.numbers),
              ),
            ),
            const SizedBox(height: 16),
            OutlinedButton(
              onPressed: _isLoading ? null : () {
                _processScan(_manualController.text);
              },
              child: _isLoading ? const CircularProgressIndicator() : const Text('Submit Manually'),
            ),
          ],
        ),
      ),
    );
  }
}
