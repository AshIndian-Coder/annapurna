import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/logistics_service.dart';
import '../../shared/widgets/profile_drawer.dart';
import '../../shared/widgets/qr_scanner_screen.dart';

class LogisticsScanScreen extends ConsumerStatefulWidget {
  const LogisticsScanScreen({super.key});

  @override
  ConsumerState<LogisticsScanScreen> createState() => _LogisticsScanScreenState();
}

class _LogisticsScanScreenState extends ConsumerState<LogisticsScanScreen> {
  final TextEditingController _manualController = TextEditingController();
  String _eventType = 'PICKED_UP';
  bool _isLoading = false;

  Future<void> _processScan(String code) async {
    final cleanCode = code.trim();
    if (_isLoading || cleanCode.isEmpty) return;
    setState(() => _isLoading = true);

    try {
      final service = ref.read(logisticsServiceProvider);
      final eventId = const Uuid().v7();
      
      final result = await service.scanEvent(
        cleanCode,
        eventType: _eventType,
        clientEventId: eventId,
      );
      
      if (!mounted) return;
      setState(() => _isLoading = false);

      result.when(
        success: (_) {
          final isPickup = _eventType == 'PICKED_UP';
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(isPickup ? 'Marked as Picked Up from Kitchen!' : 'Marked as Handed Off to NGO!'),
              backgroundColor: AppColors.good,
            ),
          );
          _manualController.clear();

          if (isPickup) {
            setState(() => _eventType = 'HANDED_OFF');
          } else {
            _showCompletionDialog();
          }
        },
        failure: (error) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Scan failed: ${error.message}'), backgroundColor: AppColors.danger),
          );
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: AppColors.danger),
      );
    }
  }

  void _showCompletionDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        icon: const Icon(Icons.check_circle, color: AppColors.good, size: 60),
        title: const Text('Delivery Completed!'),
        content: const Text(
          'Food surplus has been safely handed off and verified with the recipient NGO.',
          textAlign: TextAlign.center,
        ),
        actions: [
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
            },
            child: const Text('Done'),
          ),
        ],
      ),
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
        title: const Text('Scan & Update'),
      ),
      drawer: const ProfileDrawer(),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(value: 'PICKED_UP', label: Text('Pickup')),
                ButtonSegment(value: 'HANDED_OFF', label: Text('Drop-off')),
              ],
              selected: {_eventType},
              onSelectionChanged: (set) => setState(() => _eventType = set.first),
            ),
            const SizedBox(height: 48),
            const Icon(Icons.qr_code_scanner, size: 120, color: AppColors.textSecondary),
            const SizedBox(height: 24),
            Text(
              _eventType == 'PICKED_UP' ? 'Scan Kitchen QR' : 'Scan NGO QR',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Text(
              _eventType == 'PICKED_UP' 
                  ? 'Scan the food batch QR code at the kitchen to confirm pickup.'
                  : 'Scan the recipient QR code at the destination to confirm delivery.',
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
