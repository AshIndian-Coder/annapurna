import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/ngo_service.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/widgets/profile_drawer.dart';
import 'package:intl/intl.dart';
import '../auth/auth_provider.dart';

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

class _OfferCard extends ConsumerWidget {
  final NgoOffer offer;
  const _OfferCard({required this.offer});

  Future<void> _handleAction(BuildContext context, WidgetRef ref, bool accept) async {
    final service = ref.read(ngoServiceProvider);
    // Contract #19: POST /matches/{id}/respond — uses batchId as match identifier.
    final result = accept ? await service.acceptOffer(offer.batchId) : await service.declineOffer(offer.batchId);
    
    if (!context.mounted) return;
    
    result.when(
      success: (_) {
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

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final timeFormat = DateFormat.Hm();
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
                else
                  StatusBadge(label: offer.status),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.scale, size: 16, color: AppColors.textSecondary),
                const SizedBox(width: 8),
                Text('${offer.quantityKg} kg • ${offer.foodCategory}', style: Theme.of(context).textTheme.bodyMedium),
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
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => _handleAction(context, ref, false),
                    style: OutlinedButton.styleFrom(foregroundColor: AppColors.danger),
                    child: const Text('Decline'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: FilledButton(
                    onPressed: () => _handleAction(context, ref, true),
                    child: const Text('Accept'),
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
