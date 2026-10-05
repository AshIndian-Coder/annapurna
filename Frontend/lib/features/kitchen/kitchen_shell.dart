import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';

class KitchenShell extends ConsumerStatefulWidget {
  final Widget child;
  const KitchenShell({super.key, required this.child});

  @override
  ConsumerState<KitchenShell> createState() => _KitchenShellState();
}

class _KitchenShellState extends ConsumerState<KitchenShell> {
  int _currentIndex = 0;

  static const _routes = ['/kitchen', '/kitchen/predict', '/kitchen/surplus', '/kitchen/more'];

  @override
  Widget build(BuildContext context) {
    final location = GoRouterState.of(context).uri.toString();
    _currentIndex = _routes.lastIndexWhere((r) => location.startsWith(r));
    if (_currentIndex < 0) _currentIndex = 0;

    return Scaffold(
      body: widget.child,
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          color: AppColors.surface,
          border: Border(top: BorderSide(color: AppColors.border, width: 0.5)),
        ),
        child: NavigationBar(
          selectedIndex: _currentIndex,
          onDestinationSelected: (index) {
            if (index != _currentIndex) {
              context.go(_routes[index]);
            }
          },
          destinations: const [
            NavigationDestination(icon: Icon(Icons.dashboard_outlined), selectedIcon: Icon(Icons.dashboard), label: 'Home'),
            NavigationDestination(icon: Icon(Icons.analytics_outlined), selectedIcon: Icon(Icons.analytics), label: 'Plan'),
            NavigationDestination(icon: Icon(Icons.inventory_2_outlined), selectedIcon: Icon(Icons.inventory_2), label: 'Surplus'),
            NavigationDestination(icon: Icon(Icons.more_horiz), selectedIcon: Icon(Icons.more_horiz), label: 'More'),
          ],
        ),
      ),
    );
  }
}
