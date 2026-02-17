const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { google } = require("googleapis");

admin.initializeApp();

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

exports.syncSavedataToFirestore = functions
  .region("southamerica-east1")
  .https.onRequest(async (req, res) => {
  try {
    // ✅ 1) CONFIG: troque aqui
    const SPREADSHEET_ID = "1Z887EqYOatQ6ebMjYcjsCX4ZcTWi30F6Gf4zbCC_WZ8";
    const SHEET_NAME = "Página1"; // igual aparece no seu print

    // ✅ 2) Autenticação do Google Sheets via service account
    // Você vai colar o JSON da gcp_service_account como config do functions (passo abaixo)
    const gcpSa = functions.config().gcp_sa;
    if (!gcpSa || !gcpSa.client_email || !gcpSa.private_key) {
      return res.status(500).send("Falta configurar gcp_sa (client_email/private_key) nas functions config.");
    }

    }

    const jwt = new google.auth.JWT(
      gcpSa.client_email,
      null,
      gcpSa.private_key,
      ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    );

    const sheets = google.sheets({ version: "v4", auth: jwt });

    // ✅ 3) Lê valores A:C
    const range = `${SHEET_NAME}!A:C`;
    const resp = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range,
    });

    const rows = resp.data.values || [];
    if (rows.length <= 1) {
      return res.status(200).json({ ok: true, message: "Sem dados (apenas header?)" });
    }

    // ✅ 4) Header (linha 1) e depois dados
    const header = rows[0];
    const idxTrainer = header.indexOf("Treinador");
    const idxData = header.indexOf("Dados");

    if (idxTrainer === -1 || idxData === -1) {
      return res.status(400).send("Header precisa ter colunas 'Treinador' e 'Dados'.");
    }

    const db = admin.firestore();
    let okCount = 0;
    let failCount = 0;

    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      const trainer = (r[idxTrainer] || "").trim();
      const dataStr = r[idxData] || "";

      if (!trainer || !dataStr) continue;

      const trainerId = safeId(trainer);

      // Dados no sheet parece ser JSON como string, então tentamos parsear
      let dataObj = null;
      try {
        dataObj = JSON.parse(dataStr);
      } catch (e) {
        // se não parsear, salva como string mesmo (pra não perder)
        dataObj = { _raw: dataStr };
      }

      try {
        await db.doc(`users_raw/${trainerId}`).set(
          {
            trainer,
            trainerId,
            data: dataObj,
            updatedAt: admin.firestore.FieldValue.serverTimestamp(),
            source: "SaveData_RPG",
          },
          { merge: true }
        );
        okCount++;
      } catch (e) {
        failCount++;
      }
    }

    return res.status(200).json({ ok: true, okCount, failCount });
  } catch (err) {
    console.error(err);
    return res.status(500).send(String(err));
  }
});
