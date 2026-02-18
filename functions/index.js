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
      const SPREADSHEET_ID = "1Z887EqYOatQ6ebMjYcjsCX4ZcTWi30F6Gf4zbCC_WZ8";
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
/**
 * ✅ (3) ON-DEMAND (NÃO CLONA NO FIRESTORE)
 * getTrainerParty (southamerica-east1)
 *
 * - Consulta a planilha SaveData_RPG e devolve a party atual do treinador.
 * - Opcional: se enviar ?rid=..., também faz upsert em rooms/{rid}/players/{uid}
 *   com party_snapshot (para sincronizar a sala sem o cliente escrever no Firestore).
 *
 * Query params:
 * - trainer: string (obrigatório)
 * - rid: string (opcional)
 * - include_profile: "1"|"true" (opcional) — devolve trainer_profile (sem senha)
 */
exports.getTrainerParty = functions
  .region("southamerica-east1")
  .https
  .onRequest(async (req, res) => {
    // CORS simples (battle-site é HTML estático)
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.set("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") return res.status(204).send("");

    try {
      const trainer = String((req.query.trainer || req.body?.trainer || "")).trim();
      const rid = String((req.query.rid || req.body?.rid || "")).trim();
      const includeProfile = String((req.query.include_profile || req.body?.include_profile || "")).toLowerCase();
      const wantProfile = includeProfile === "1" || includeProfile === "true" || includeProfile === "yes";

      if (!trainer) return res.status(400).json({ ok: false, error: "missing trainer" });

      // CONFIG: use o mesmo Spreadsheet/Sheet da sua sync function
      const SPREADSHEET_ID = "1Z887EqYOatQ6ebMjYcjsCX4ZcTWi30F6Gf4zbCC_WZ8";
      const SHEET_NAME = "Página1";

      const gcpSa = functions.config().gcp_sa;
      if (!gcpSa || !gcpSa.client_email || !gcpSa.private_key) {
        return res.status(500).json({ ok: false, error: "missing gcp_sa functions config" });
      }
      const privateKey = String(gcpSa.private_key).replace(/\\n/g, "\n");

      const jwt = new google.auth.JWT(
        gcpSa.client_email,
        null,
        privateKey,
        ["https://www.googleapis.com/auth/spreadsheets.readonly"]
      );
      const sheets = google.sheets({ version: "v4", auth: jwt });

      // Lê A:C (suporta ambos headers: Treinador/Dados/(Senha) OU Nome/JSON/(Senha))
      const range = `${SHEET_NAME}!A:C`;
      const resp = await sheets.spreadsheets.values.get({ spreadsheetId: SPREADSHEET_ID, range });
      const rows = resp.data.values || [];
      if (!rows.length) return res.status(200).json({ ok: false, error: "empty sheet" });

      const header = rows[0].map((x) => String(x || "").trim());
      const low = header.map((x) => x.toLowerCase());

      const idxTrainer = low.indexOf("treinador") !== -1 ? low.indexOf("treinador") : low.indexOf("nome");
      const idxData = low.indexOf("dados") !== -1 ? low.indexOf("dados") : low.indexOf("json");

      // Se não tem header, assume A=Nome, B=JSON, C=Senha
      const hasHeader = idxTrainer !== -1 && idxData !== -1;
      const start = hasHeader ? 1 : 0;
      const iTrainer = hasHeader ? idxTrainer : 0;
      const iData = hasHeader ? idxData : 1;

      const norm = (s) =>
        String(s || "")
          .trim()
          .toLowerCase()
          .normalize("NFD").replace(/[\u0300-\u036f]/g, "");

      const target = norm(trainer);
      let found = null;

      for (let i = start; i < rows.length; i++) {
        const r = rows[i] || [];
        const nameCell = r[iTrainer];
        if (!nameCell) continue;
        if (norm(nameCell) === target) {
          found = r;
          break;
        }
      }

      if (!found) {
        return res.status(404).json({ ok: false, error: "trainer not found", trainer });
      }

      const dataStr = String(found[iData] || "");
      let dataObj = null;
      try {
        dataObj = JSON.parse(dataStr);
      } catch {
        dataObj = null;
      }

      const party = Array.isArray(dataObj?.party) ? dataObj.party : [];
      // padroniza para strings (IDs da sua Dex)
      const partySnapshot = party.map((pid) => ({ pid: String(pid) }));

      const trainerId = safeId(trainer) || safeId(trainer); // mantém a mesma regra do projeto
      const out = {
        ok: true,
        trainer,
        uid: trainerId,
        party: partySnapshot,
        updatedAt: new Date().toISOString(),
      };

      if (wantProfile) {
        out.trainer_profile = (dataObj && typeof dataObj.trainer_profile === "object") ? dataObj.trainer_profile : null;
      }

      // Opcional: espelha SOMENTE party_snapshot para a sala (não copia SaveData inteira)
      if (rid) {
        const pRef = db.doc(`rooms/${rid}/players/${trainerId}`);
        await pRef.set(
          {
            uid: trainerId,
            trainer_name: trainer,
            party_snapshot: partySnapshot,
            updatedAt: TS,
            source: "SaveData_RPG",
          },
          { merge: true }
        );
      }

      return res.status(200).json(out);
    } catch (err) {
      functions.logger.error("getTrainerParty_ERROR", err);
      return res.status(500).json({ ok: false, error: String(err?.message || err) });
    }
  });
