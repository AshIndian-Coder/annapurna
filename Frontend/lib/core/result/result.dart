import '../errors/app_error.dart';

sealed class Result<T> {
  const Result();

  bool get isSuccess => this is Success<T>;
  bool get isFailure => this is Failure<T>;

  T? get dataOrNull => switch (this) {
    Success<T> s => s.data,
    Failure<T> _ => null,
  };

  AppError? get errorOrNull => switch (this) {
    Success<T> _ => null,
    Failure<T> f => f.error,
  };

  R when<R>({
    required R Function(T data) success,
    required R Function(AppError error) failure,
  }) {
    return switch (this) {
      Success<T> s => success(s.data),
      Failure<T> f => failure(f.error),
    };
  }
}

class Success<T> extends Result<T> {
  final T data;
  const Success(this.data);
}

class Failure<T> extends Result<T> {
  final AppError error;
  const Failure(this.error);
}
