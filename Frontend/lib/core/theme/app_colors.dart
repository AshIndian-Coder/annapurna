import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  // Primary: Blue
  static const Color primary = Color(0xFF3B82F6);
  static const Color primaryDark = Color(0xFF2563EB);
  static const Color primaryLight = Color(0xFF60A5FA);
  static const Color primarySurface = Color(0xFF1E3A5F);

  // Accent: Yellow
  static const Color accent = Color(0xFFF59E0B);
  static const Color accentDark = Color(0xFFD97706);
  static const Color accentLight = Color(0xFFFBBF24);

  // Background: Pure Black
  static const Color background = Color(0xFF000000);
  static const Color surface = Color(0xFF111111);
  static const Color surfaceElevated = Color(0xFF1C1C1C);
  static const Color surfaceHighlight = Color(0xFF2A2A2A);

  // Text
  static const Color textPrimary = Color(0xFFF1F5F9);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);
  static const Color textOnPrimary = Color(0xFFFFFFFF);

  // Status: Blue + Yellow only, no green
  static const Color good = Color(0xFF3B82F6);       // Blue (was green)
  static const Color warning = Color(0xFFF59E0B);     // Yellow
  static const Color danger = Color(0xFFEF4444);       // Red (kept for errors)
  static const Color info = Color(0xFF60A5FA);          // Light blue
  static const Color hold = Color(0xFFFBBF24);          // Yellow (was purple)

  // Borders
  static const Color border = Color(0xFF1C1C1C);
  static const Color borderLight = Color(0xFF2A2A2A);
  static const Color divider = Color(0xFF111111);

  // Shimmer
  static const Color shimmerBase = Color(0xFF111111);
  static const Color shimmerHighlight = Color(0xFF1C1C1C);

  // Gradients: Blue + Yellow
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [Color(0xFF3B82F6), Color(0xFF2563EB)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient accentGradient = LinearGradient(
    colors: [Color(0xFFF59E0B), Color(0xFFD97706)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient dangerGradient = LinearGradient(
    colors: [Color(0xFFEF4444), Color(0xFFDC2626)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient surfaceGradient = LinearGradient(
    colors: [Color(0xFF111111), Color(0xFF000000)],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static Color statusColor(String status) {
    return switch (status) {
      'AVAILABLE' || 'ELIGIBLE' || 'GOOD' || 'DELIVERED' || 'ACCEPTED' => good,
      'PENDING_SAFETY' || 'HOLD' || 'UNKNOWN' || 'MEDIUM' => hold,
      'MATCHED' || 'IN_TRANSIT' || 'LOW' => info,
      'EXPIRED' || 'REJECTED' || 'DIVERTED' || 'RISK' || 'HIGH' || 'CRITICAL' => danger,
      'WARN' || 'EXPIRY_SOON' => warning,
      _ => textSecondary,
    };
  }
}
