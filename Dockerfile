# ─── Stage 1: Build ───────────────────────────────────────────────────────────
FROM golang:1.23-alpine AS builder

# Install git and CA certs (needed by go mod download for private/public deps)
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /build

# Cache dependency downloads before copying full source
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the source tree
COPY . .

# Build a statically-linked binary (CGO disabled for distroless compatibility)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
      -trimpath \
      -ldflags="-s -w -extldflags '-static'" \
      -o /build/server \
      ./cmd/server

# ─── Stage 2: Final (distroless) ──────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot

# Copy timezone data and CA certs from builder so HTTPS + TZ work at runtime
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the compiled binary
COPY --from=builder /build/server /server

# The nonroot image already runs as uid=65532; no USER directive needed.
EXPOSE 8080

ENTRYPOINT ["/server"]
