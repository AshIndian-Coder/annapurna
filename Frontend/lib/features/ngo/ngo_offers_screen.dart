import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:qr_flutter/qr_flutter.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/ngo_service.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/widgets/profile_drawer.dart';
import 'package:intl/intl.dart';

final ngoOffersProvider = FutureProvider.autoDispose<List<NgoOffer>>((ref) async {
  final service = ref.read(ngoServiceProvider);
  final result = await service.getOffers();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class NgoOffersScreen extends ConsumerWidget {
  const NgoOffersScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final offersAsync = ref.watch(ngoOffersProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Available Offers'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.refresh(ngoOffersProvider),
          ),
        ],
      ),
      drawer: const ProfileDrawer(),
      body: offersAsync.when(
        loading: () => ListView.separated(
          padding: const EdgeInsets.all(16),
          itemCount: 4,
          separatorBuilder: (_, __) => const SizedBox(height: 12),
          itemBuilder: (_, __) => const SkeletonLoader(height: 160, borderRadius: 16),
        ),
        error: (err, stack) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(ngoOffersProvider),
        ),
        data: (offers) {
          if (offers.isEmpty) {
            return const EmptyState(
              icon: Icons.inbox_outlined,
              title: 'No offers right now',
              subtitle: 'We will notify you when surplus food is available nearby.',
            );
          }
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(ngoOffersProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: offers.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) => _OfferCard(offer: offers[index]),
            ),
          );
        },
      ),
    );
  }
}

class _OfferCard extends ConsumerStatefulWidget {
  final NgoOffer offer;
  const _OfferCard({required this.offer});

  @override
  ConsumerState<_OfferCard> createState() => _OfferCardState();
}

class _OfferCardState extends ConsumerState<_OfferCard> {
  bool _isLocallyAccepted = false;
  bool _isLoading = false;

  Future<void> _handleAction(BuildContext context, bool accept) async {
    setState(() => _isLoading = true);
    final service = ref.read(ngoServiceProvider);
    final result = accept 
        ? await service.acceptOffer(widget.offer.batchId) 
        : await service.declineOffer(widget.offer.batchId);
    
    if (!mounted) return;
    setState(() => _isLoading = false);

    result.when(
      success: (_) {
        if (accept) {
          setState(() => _isLocallyAccepted = true);
        }
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(accept ? 'Offer accepted' : 'Offer declined')),
        );
        ref.refresh(ngoOffersProvider);
      },
      failure: (error) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(error.message), backgroundColor: AppColors.danger),
        );
      },
    );
  }

  void _showDeliveryQrDialog(BuildContext context) {
    final offer = widget.offer;
    final manualCode = offer.batchCode.isNotEmpty ? offer.batchCode : offer.batchId;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        title: Text('Delivery Handoff QR - ${offer.foodName}'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Show this QR code to the delivery driver upon arrival.',
                style: TextStyle(color: AppColors.textSecondary, fontSize: 13),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: QrImageView(
                  data: offer.batchId,
                  version: QrVersions.auto,
                  size: 200.0,
                  backgroundColor: Colors.white,
                ),
              ),
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  children: [
                    const Text(
                      'If QR scanner does not work, use Batch Number:',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 12, color: AppColors.textSecondary),
                    ),
                    const SizedBox(height: 6),
                    SelectableText(
                      manualCode,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.5,
                        fontFamily: 'monospace',
                        color: AppColors.primary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final offer = widget.offer;
    final timeFormat = DateFormat.Hm();
    final bool isAccepted = _isLocallyAccepted || 
        offer.status.toUpperCase() == 'ACCEPTED' || 
        offer.status.toUpperCase() == 'MATCHED' || 
        offer.status.toUpperCase() == 'IN_TRANSIT';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    offer.foodName,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ),
                if (offer.isUrgent)
                  const StatusBadge(label: 'URGENT')
                else if (isAccepted)
                  const StatusBadge(label: 'ACCEPTED')
                else
                  StatusBadge(label: offer.status),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.scale, size: 16, color: AppColors.textSecondary),
                const SizedBox(width: 8),
                Text('${offer.quantityKg} ${offer.quantityUnit} • ${offer.foodCategory}', style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.qr_code, size: 16, color: AppColors.textSecondary),
                const SizedBox(width: 8),
                Text('Batch: ${offer.batchCode}', style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.access_time, size: 16, color: AppColors.textSecondary),
                const SizedBox(width: 8),
                Text('Expires ${timeFormat.format(offer.expiryAt)}', style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
            const SizedBox(height: 16),
            if (isAccepted)
              Wrap(
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 10,
                runSpacing: 10,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: AppColors.good.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: AppColors.good.withOpacity(0.4)),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.check_circle, size: 16, color: AppColors.good),
                        SizedBox(width: 6),
                        Text(
                          'Accepted',
                          style: TextStyle(color: AppColors.good, fontWeight: FontWeight.w600, fontSize: 13),
                        ),
                      ],
                    ),
                  ),
                  FilledButton.icon(
                    onPressed: () => _showDeliveryQrDialog(context),
                    icon: const Icon(Icons.qr_code, size: 18),
                    label: const Text('Show Delivery QR'),
                    style: FilledButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    ),
                  ),
                ],
              )
            else
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: _isLoading ? null : () => _handleAction(context, false),
                      style: OutlinedButton.styleFrom(foregroundColor: AppColors.danger),
                      child: const Text('Decline'),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: FilledButton(
                      onPressed: _isLoading ? null : () => _handleAction(context, true),
                      child: _isLoading 
                          ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Text('Accept'),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
