import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';

class LogisticsShell extends StatefulWidget {
  final Widget child;
  const LogisticsShell({super.key, required this.child});

  @override
  State<LogisticsShell> createState() => _LogisticsShellState();
}

class _LogisticsShellState extends State<LogisticsShell> {
  int _currentIndex = 0;

  static const _routes = ['/logistics', '/logistics/scan'];

  @override
  Widget build(BuildContext context) {
    final location = GoRouterState.of(context).uri.toString();
    _currentIndex = _routes.indexWhere((r) => location == r || (location.startsWith(r) && r != '/logistics'));
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
            NavigationDestination(icon: Icon(Icons.route_outlined), selectedIcon: Icon(Icons.route), label: 'Today'),
            NavigationDestination(icon: Icon(Icons.qr_code_scanner_outlined), selectedIcon: Icon(Icons.qr_code_scanner), label: 'Scan'),
          ],
        ),
      ),
    );
  }
}
