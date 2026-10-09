import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';

class NgoShell extends StatefulWidget {
  final Widget child;
  const NgoShell({super.key, required this.child});

  @override
  State<NgoShell> createState() => _NgoShellState();
}

class _NgoShellState extends State<NgoShell> {
  int _currentIndex = 0;

  static const _routes = ['/ngo', '/ngo/history'];

  @override
  Widget build(BuildContext context) {
    final location = GoRouterState.of(context).uri.toString();
    _currentIndex = _routes.indexWhere((r) => location == r || (location.startsWith(r) && r != '/ngo'));
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
            NavigationDestination(icon: Icon(Icons.local_offer_outlined), selectedIcon: Icon(Icons.local_offer), label: 'Offers'),
            NavigationDestination(icon: Icon(Icons.history_outlined), selectedIcon: Icon(Icons.history), label: 'History'),
          ],
        ),
      ),
    );
  }
}
