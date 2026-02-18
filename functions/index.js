const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { google } = require("googleapis");

admin.initializeApp();
const db = admin.firestore();
const TS = admin.firestore.FieldValue.serverTimestamp();

/**
 * Converte nome do treinador em id seguro do Firestore
 */
function safeId(name) {
  return String(name || "")
    .trim()
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "") // tira acentos
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

/**
 * ✅ (1) SUA FUNCTION EXISTENTE - NÃO REMOVER
 * applyRoomAction (us-central1)
 */
exports.applyRoomAction = functions
  .region("us-central1")
  .firestore
  .document("rooms/{rid}/actions/{actionId}")
  .onCreate(async (snap, context) => {
    const { rid, actionId } = context.params;
    const action = snap.data() || {};

    functions.logger.info("ACTION_RECEIVED", { rid, actionId, raw: action });

    const type = String(action.type || "").toUpperCase();
    const by = String(action.by || "unknown");
    const payload = action.payload || {};

    functions.logger.info("ACTION_PARSED", { rid, actionId, type, by, payload });

    const stateRef = db.doc(`rooms/${rid}/public_state/state`);
    const battleRef = db.doc(`rooms/${rid}/public_state/battle`);
    const actionRef = db.doc(`rooms/${rid}/actions/${actionId}`);

    try {
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

      await actionRef.set({ status: "ignored", reason: "unknown action type" }, { merge: true });
      return null;

    } catch (e) {
      await actionRef.set({ status: "error", error: String(e?.message || e) }, { merge: true });
      return null;
    }
  });

/**
 * ✅ (2) SUA NOVA FUNCTION
 * syncSavedataToFirestore (southamerica-east1)
 */
exports.syncSavedataToFirestore = functions
  .region("southamerica-east1")
  .https
  .onRequest(async (req, res) => {
    try {
      // (opcional) você pode travar só GET/POST:
      // if (!["GET", "POST"].includes(req.method)) return res.status(405).send("Method Not Allowed");

      // ✅ 1) CONFIG
      const SPREADSHEET_ID = "COLE_AQUI_O_ID_DA_PLANILHA";
      const SHEET_NAME = "Página1";

      // ✅ 2) Autenticação: functions config
      const gcpSa = functions.config().gcp_sa;
      if (!gcpSa || !gcpSa.client_email || !gcpSa.private_key) {
        return res.status(500).send("Falta configurar gcp_sa (client_email/private_key) nas functions config.");
      }

      // 🔥 MUITO comum a private_key vir com \\n em vez de \n
      const privateKey = String(gcpSa.private_key).replace(/\\n/g, "\n");

      const jwt = new google.auth.JWT(
        gcpSa.client_email,
        null,
        privateKey,
        ["https://www.googleapis.com/auth/spreadsheets.readonly"]
      );

      const sheets = google.sheets({ version: "v4", auth: jwt });

      // ✅ 3) Lê A:C
      const range = `${SHEET_NAME}!A:C`;
      const resp = await sheets.spreadsheets.values.get({ spreadsheetId: SPREADSHEET_ID, range });

      const rows = resp.data.values || [];
      if (rows.length <= 1) {
        return res.status(200).json({ ok: true, message: "Sem dados (apenas header?)" });
      }

      // ✅ 4) Header
      const header = rows[0];
      const idxTrainer = header.indexOf("Treinador");
      const idxData = header.indexOf("Dados");

      if (idxTrainer === -1 || idxData === -1) {
        return res.status(400).send("Header precisa ter colunas 'Treinador' e 'Dados'.");
      }

      let okCount = 0;
      let failCount = 0;

      for (let i = 1; i < rows.length; i++) {
        const r = rows[i];
        const trainer = (r[idxTrainer] || "").trim();
        const dataStr = r[idxData] || "";

        if (!trainer || !dataStr) continue;

        const trainerId = safeId(trainer);

        let dataObj;
        try {
          dataObj = JSON.parse(dataStr);
        } catch {
          dataObj = { _raw: dataStr };
        }

        try {
          await db.doc(`users_raw/${trainerId}`).set(
            {
              trainer,
              trainerId,
              data: dataObj,
              updatedAt: TS,
              source: "SaveData_RPG",
            },
            { merge: true }
          );
          okCount++;
        } catch (e) {
          failCount++;
          functions.logger.error("WRITE_FAIL", { trainer, trainerId, err: String(e?.message || e) });
        }
      }

      return res.status(200).json({ ok: true, okCount, failCount });
    } catch (err) {
      functions.logger.error("syncSavedataToFirestore_ERROR", err);
      return res.status(500).send(String(err?.message || err));
    }
  });
