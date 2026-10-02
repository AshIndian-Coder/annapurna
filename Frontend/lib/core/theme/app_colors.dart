import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  static const Color primary = Color(0xFF10B981);
  static const Color primaryDark = Color(0xFF059669);
  static const Color primaryLight = Color(0xFF34D399);
  static const Color primarySurface = Color(0xFF064E3B);

  static const Color accent = Color(0xFFF59E0B);
  static const Color accentDark = Color(0xFFD97706);
  static const Color accentLight = Color(0xFFFBBF24);

  static const Color background = Color(0xFF0F172A);
  static const Color surface = Color(0xFF1E293B);
  static const Color surfaceElevated = Color(0xFF334155);
  static const Color surfaceHighlight = Color(0xFF475569);

  static const Color textPrimary = Color(0xFFF1F5F9);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);
  static const Color textOnPrimary = Color(0xFFFFFFFF);

  static const Color good = Color(0xFF10B981);
  static const Color warning = Color(0xFFF59E0B);
  static const Color danger = Color(0xFFEF4444);
  static const Color info = Color(0xFF3B82F6);
  static const Color hold = Color(0xFF8B5CF6);

  static const Color border = Color(0xFF334155);
  static const Color borderLight = Color(0xFF475569);
  static const Color divider = Color(0xFF1E293B);

  static const Color shimmerBase = Color(0xFF1E293B);
  static const Color shimmerHighlight = Color(0xFF334155);

  static const LinearGradient primaryGradient = LinearGradient(
    colors: [Color(0xFF10B981), Color(0xFF059669)],
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
    colors: [Color(0xFF1E293B), Color(0xFF0F172A)],
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
