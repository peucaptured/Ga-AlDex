const functions = require("firebase-functions");
const admin = require("firebase-admin");

admin.initializeApp();
const db = admin.firestore();
const TS = admin.firestore.FieldValue.serverTimestamp();

exports.applyRoomAction = functions.firestore
  .document("rooms/{rid}/actions/{actionId}")
  .onCreate(async (snap, context) => {
    const { rid, actionId } = context.params;
    const action = snap.data() || {};

    const type = String(action.type || "").toUpperCase();
    const by = String(action.by || "unknown");
    const payload = action.payload || {};

    const stateRef = db.doc(`rooms/${rid}/public_state/state`);
    const battleRef = db.doc(`rooms/${rid}/public_state/battle`);
    const actionRef = db.doc(`rooms/${rid}/actions/${actionId}`);

    try {
      // ADD_LOG: adiciona em battle.logs
      if (type === "ADD_LOG") {
        await db.runTransaction(async (tx) => {
          const bSnap = await tx.get(battleRef);
          const battle = bSnap.exists ? bSnap.data() : {};
          const logs = Array.isArray(battle.logs) ? battle.logs : [];

          logs.push({
            t: "log",
            by,
            text: String(payload.text || ""),
            at: admin.firestore.Timestamp.now(),
          });

          tx.set(battleRef, { logs, updatedAt: TS }, { merge: true });
          tx.set(actionRef, { status: "applied", appliedAt: TS }, { merge: true });
        });
        return null;
      }

      // MOVE_PIECE: move peça por pieceId e row/col
      if (type === "MOVE_PIECE") {
        const pieceId = String(payload.pieceId || "");
        const row = Number(payload.row);
        const col = Number(payload.col);

        if (!pieceId || !Number.isFinite(row) || !Number.isFinite(col)) {
          await actionRef.set({ status: "rejected", reason: "invalid payload" }, { merge: true });
          return null;
        }

        await db.runTransaction(async (tx) => {
          const sSnap = await tx.get(stateRef);
          const state = sSnap.exists ? sSnap.data() : {};

          // Suporta state.pieces (lista) OU state.piecesById (mapa)
          if (Array.isArray(state.pieces)) {
            const pieces = state.pieces.map((p) => {
              if (String(p.id) === pieceId) return { ...p, row, col };
              return p;
            });
            tx.set(stateRef, { pieces, updatedAt: TS }, { merge: true });
          } else if (state.piecesById && typeof state.piecesById === "object") {
            const piecesById = { ...state.piecesById };
            if (!piecesById[pieceId]) {
              tx.set(actionRef, { status: "rejected", reason: "piece not found" }, { merge: true });
              return;
            }
            piecesById[pieceId] = { ...piecesById[pieceId], row, col };
            tx.set(stateRef, { piecesById, updatedAt: TS }, { merge: true });
          } else {
            tx.set(actionRef, { status: "rejected", reason: "no pieces in state" }, { merge: true });
            return;
          }

          tx.set(actionRef, { status: "applied", appliedAt: TS }, { merge: true });
        });

        return null;
      }

      // Não implementado ainda
      await actionRef.set({ status: "ignored", reason: "unknown action type" }, { merge: true });
      return null;

    } catch (e) {
      await actionRef.set({ status: "error", error: String(e?.message || e) }, { merge: true });
      return null;
    }
  });
