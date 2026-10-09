// Package qrchain implements the canonical SHA-256 hash chain for QR code
// events in the SIH26234 food-waste redistribution platform.
//
// Chain integrity rule (per spec D30):
//
//	event_hash = SHA-256( prevHash || batchID || eventType || actorID || serverTSiso || evidenceHash )
//
// Fields are concatenated as raw ASCII/UTF-8 bytes with NO separator bytes
// between them.  The genesis event uses the constant "GENESIS" as prevHash.
package qrchain

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// GenesisHash is the prevHash value used for the very first event in any
// batch's QR chain.
const GenesisHash = "GENESIS"

// QREvent is the minimal view of a QR chain event required for verification.
// It must match the qr_events row in creation order.
type QREvent struct {
	// PrevHash is the event_hash of the immediately preceding event, or
	// GenesisHash for the first event.
	PrevHash string
	// EventHash is the stored hash that we are verifying.
	EventHash string

	// The five fields that were hashed to produce EventHash:
	BatchID      string // UUID string representation
	EventType    string
	ActorID      string // UUID string representation
	ServerTSIso  string // RFC-3339 / ISO-8601 timestamp as stored
	EvidenceHash string // empty string if no evidence
}

// Hash computes the canonical event hash.
//
// It concatenates the six input strings as raw bytes (no separators) and
// returns the lower-case hex-encoded SHA-256 digest.
//
//	hash = hex( SHA-256( prevHash + batchID + eventType + actorID + serverTSiso + evidenceHash ) )
func Hash(prevHash, batchID, eventType, actorID, serverTSiso, evidenceHash string) string {
	h := sha256.New()
	// Write returns a nil error for hash.Hash implementations per the spec.
	_, _ = fmt.Fprint(h, prevHash)
	_, _ = fmt.Fprint(h, batchID)
	_, _ = fmt.Fprint(h, eventType)
	_, _ = fmt.Fprint(h, actorID)
	_, _ = fmt.Fprint(h, serverTSiso)
	_, _ = fmt.Fprint(h, evidenceHash)
	return hex.EncodeToString(h.Sum(nil))
}

// EvidenceHash computes the SHA-256 hash of a serialised quality-check JSON
// blob.  The result is embedded in the QR event that references the check.
// Pass an empty slice to get the zero-evidence hash ("").
func EvidenceHash(qualityCheckJSON []byte) string {
	if len(qualityCheckJSON) == 0 {
		return ""
	}
	sum := sha256.Sum256(qualityCheckJSON)
	return hex.EncodeToString(sum[:])
}

// VerifyChain walks events in the order provided (expected: ascending
// created_at) and re-computes each event_hash from the stored fields.
//
// Returns:
//   - valid=true, brokenAt=-1  → chain is intact
//   - valid=false, brokenAt=i  → event at index i has a mismatching hash
//
// The first event in the slice must have PrevHash == GenesisHash; subsequent
// events must have PrevHash == EventHash of the previous event.
func VerifyChain(events []QREvent) (valid bool, brokenAt int) {
	for i, e := range events {
		expected := Hash(
			e.PrevHash,
			e.BatchID,
			e.EventType,
			e.ActorID,
			e.ServerTSIso,
			e.EvidenceHash,
		)
		if expected != e.EventHash {
			return false, i
		}
		// Verify the linkage: from the second event onwards, prevHash must
		// equal the EventHash of the previous event.
		if i > 0 && e.PrevHash != events[i-1].EventHash {
			return false, i
		}
	}
	return true, -1
}
