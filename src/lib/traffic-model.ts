// Smart Traffic Density Prediction — Multinomial Logistic Regression
// Faithful port of the reference Jupyter notebook:
//   Features: Vehicle_Count, Average_Speed, Signal_Time, Road_Capacity
//   Target:   Traffic_Level = cut(Vehicle_Count - Average_Speed + Signal_Time,
//                                 bins=[0,200,400,1000], labels=[Low,Medium,High])
//   Pipeline: StandardScaler -> LogisticRegression(max_iter=1000)
//   Dataset:  50 rows, np.random.seed(42) -> reproduced here with an LCG seeded to match
//             the *shape* of the notebook (identical formula & bin edges).
// All training happens in-browser on load — no server required.

export type TrafficClass = "Low" | "Medium" | "High";
export const CLASSES: TrafficClass[] = ["High", "Low", "Medium"]; // LabelEncoder alpha-order: High=0, Low=1, Medium=2

export type TrafficRow = {
  Vehicle_Count: number;
  Average_Speed: number;
  Signal_Time: number;
  Road_Capacity: number;
  Traffic_Level: TrafficClass;
  y: number; // encoded label 0..2
};

// ---------- deterministic PRNG (mulberry32) ----------
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randInt(rng: () => number, lo: number, hi: number) {
  // matches np.random.randint(lo, hi) — exclusive upper bound
  return lo + Math.floor(rng() * (hi - lo));
}

function labelFor(vc: number, sp: number, st: number): TrafficClass {
  const s = vc - sp + st;
  if (s > 0 && s <= 200) return "Low";
  if (s > 200 && s <= 400) return "Medium";
  if (s > 400) return "High";
  return "Low";
}
const CLASS_IDX: Record<TrafficClass, number> = { High: 0, Low: 1, Medium: 2 };

// ---------- build the dataset ----------
function buildDataset(n = 50, seed = 42): TrafficRow[] {
  const rng = mulberry32(seed);
  const rows: TrafficRow[] = [];
  for (let i = 0; i < n; i++) {
    const vc = randInt(rng, 50, 500);
    const sp = randInt(rng, 10, 80);
    const st = randInt(rng, 20, 120);
    const rc = randInt(rng, 100, 600);
    const lvl = labelFor(vc, sp, st);
    rows.push({
      Vehicle_Count: vc, Average_Speed: sp, Signal_Time: st, Road_Capacity: rc,
      Traffic_Level: lvl, y: CLASS_IDX[lvl],
    });
  }
  return rows;
}

// ---------- StandardScaler ----------
type Scaler = { mean: number[]; std: number[] };
function fitScaler(X: number[][]): Scaler {
  const d = X[0].length, n = X.length;
  const mean = Array(d).fill(0), std = Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (const row of X) for (let j = 0; j < d; j++) std[j] += (row[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1;
  return { mean, std };
}
function transform(X: number[][], s: Scaler): number[][] {
  return X.map(r => r.map((v, j) => (v - s.mean[j]) / s.std[j]));
}

// ---------- Multinomial Logistic Regression ----------
type Model = { W: number[][]; b: number[]; K: number; D: number };
function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const ex = z.map(v => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / s);
}
function trainLR(X: number[][], y: number[], K: number, iters = 1500, lr = 0.15, l2 = 0.01): Model {
  const n = X.length, D = X[0].length;
  const W: number[][] = Array.from({ length: K }, () => Array(D).fill(0));
  const b: number[] = Array(K).fill(0);
  for (let it = 0; it < iters; it++) {
    const gW: number[][] = Array.from({ length: K }, () => Array(D).fill(0));
    const gb: number[] = Array(K).fill(0);
    for (let i = 0; i < n; i++) {
      const z = Array(K).fill(0);
      for (let k = 0; k < K; k++) {
        let s = b[k];
        for (let j = 0; j < D; j++) s += W[k][j] * X[i][j];
        z[k] = s;
      }
      const p = softmax(z);
      for (let k = 0; k < K; k++) {
        const err = p[k] - (y[i] === k ? 1 : 0);
        gb[k] += err;
        for (let j = 0; j < D; j++) gW[k][j] += err * X[i][j];
      }
    }
    for (let k = 0; k < K; k++) {
      b[k] -= (lr / n) * gb[k];
      for (let j = 0; j < D; j++) W[k][j] -= (lr / n) * (gW[k][j] + l2 * W[k][j]);
    }
  }
  return { W, b, K, D };
}
function predictProba(m: Model, x: number[]): number[] {
  const z = Array(m.K).fill(0);
  for (let k = 0; k < m.K; k++) {
    let s = m.b[k];
    for (let j = 0; j < m.D; j++) s += m.W[k][j] * x[j];
    z[k] = s;
  }
  return softmax(z);
}
function predictOne(m: Model, x: number[]): number {
  const p = predictProba(m, x);
  let best = 0;
  for (let k = 1; k < m.K; k++) if (p[k] > p[best]) best = k;
  return best;
}

// ---------- 80/20 split (deterministic) ----------
function split<T>(arr: T[], frac = 0.8, seed = 7): { train: T[]; test: T[]; testIdx: number[] } {
  const rng = mulberry32(seed);
  const idx = arr.map((_, i) => i).sort(() => rng() - 0.5);
  const cut = Math.floor(arr.length * frac);
  const trainI = idx.slice(0, cut), testI = idx.slice(cut);
  return { train: trainI.map(i => arr[i]), test: testI.map(i => arr[i]), testIdx: testI };
}

// ---------- build everything once ----------
const FEATURES = ["Vehicle_Count", "Average_Speed", "Signal_Time", "Road_Capacity"] as const;
const DATA = buildDataset(50, 42);
const { train: TRAIN, test: TEST } = split(DATA, 0.8, 7);

const X_train_raw = TRAIN.map(r => [r.Vehicle_Count, r.Average_Speed, r.Signal_Time, r.Road_Capacity]);
const y_train = TRAIN.map(r => r.y);
const X_test_raw = TEST.map(r => [r.Vehicle_Count, r.Average_Speed, r.Signal_Time, r.Road_Capacity]);
const y_test = TEST.map(r => r.y);

const SCALER = fitScaler(X_train_raw);
const X_train = transform(X_train_raw, SCALER);
const X_test = transform(X_test_raw, SCALER);
const MODEL = trainLR(X_train, y_train, CLASSES.length);

// ---------- metrics ----------
const y_pred = X_test.map(x => predictOne(MODEL, x));
const y_prob = X_test.map(x => predictProba(MODEL, x));

function confusion(yt: number[], yp: number[], K: number) {
  const m: number[][] = Array.from({ length: K }, () => Array(K).fill(0));
  for (let i = 0; i < yt.length; i++) m[yt[i]][yp[i]]++;
  return m;
}
const CM = confusion(y_test, y_pred, CLASSES.length);
const ACC = y_test.length ? y_test.reduce((a, v, i) => a + (v === y_pred[i] ? 1 : 0), 0) / y_test.length : 0;

function prf(cls: number) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < y_test.length; i++) {
    if (y_pred[i] === cls && y_test[i] === cls) tp++;
    else if (y_pred[i] === cls && y_test[i] !== cls) fp++;
    else if (y_pred[i] !== cls && y_test[i] === cls) fn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const support = tp + fn;
  return { precision, recall, f1, support };
}

// One-vs-rest ROC per class
function rocCurve(scores: number[], labels: number[]) {
  const pts: { fpr: number; tpr: number; thr: number }[] = [];
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = labels.filter(v => v === 1).length;
  const N = labels.length - P;
  let tp = 0, fp = 0;
  pts.push({ fpr: 0, tpr: 0, thr: Infinity });
  for (const i of order) {
    if (labels[i] === 1) tp++; else fp++;
    pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0, thr: scores[i] });
  }
  return pts;
}
function auc(pts: { fpr: number; tpr: number }[]) {
  let a = 0;
  for (let i = 1; i < pts.length; i++) {
    a += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
  }
  return a;
}

const ROC_PER_CLASS = CLASSES.map((_, k) => {
  const scores = y_prob.map(p => p[k]);
  const labels = y_test.map(v => (v === k ? 1 : 0));
  const pts = rocCurve(scores, labels);
  return { cls: CLASSES[k], points: pts, auc: auc(pts) };
});

// Feature importance = mean abs standardized coefficient across classes
const FEATURE_IMPORTANCE_ARR = FEATURES.map((f, j) => {
  let s = 0;
  for (let k = 0; k < MODEL.K; k++) s += Math.abs(MODEL.W[k][j]);
  return { feature: f.replace("_", " "), weight: +(s / MODEL.K).toFixed(3) };
}).sort((a, b) => b.weight - a.weight);

// ---------- public API ----------
export type PredictionInput = {
  Vehicle_Count: number;
  Average_Speed: number;
  Signal_Time: number;
  Road_Capacity: number;
};

export function predictTraffic(x: PredictionInput): {
  label: TrafficClass;
  probs: { cls: TrafficClass; p: number }[];
  topProb: number;
} {
  const raw = [x.Vehicle_Count, x.Average_Speed, x.Signal_Time, x.Road_Capacity];
  const scaled = raw.map((v, j) => (v - SCALER.mean[j]) / SCALER.std[j]);
  const p = predictProba(MODEL, scaled);
  const probs = p.map((pp, k) => ({ cls: CLASSES[k], p: pp }));
  let bestK = 0;
  for (let k = 1; k < p.length; k++) if (p[k] > p[bestK]) bestK = k;
  return { label: CLASSES[bestK], probs, topProb: p[bestK] };
}

export const DATASET = DATA;
export const TRAIN_SIZE = TRAIN.length;
export const TEST_SIZE = TEST.length;

export const MODEL_METRICS = {
  accuracy: ACC,
  perClass: CLASSES.map((c, k) => ({ cls: c, ...prf(k) })),
  confusionMatrix: CM,
  classes: CLASSES,
};

export const ROC_CURVES = ROC_PER_CLASS;
export const FEATURE_IMPORTANCE = FEATURE_IMPORTANCE_ARR;

// Aggregate patterns for dashboard
export const CLASS_DISTRIBUTION = (() => {
  const counts: Record<TrafficClass, number> = { Low: 0, Medium: 0, High: 0 };
  for (const r of DATA) counts[r.Traffic_Level]++;
  return (["Low", "Medium", "High"] as TrafficClass[]).map(c => ({ cls: c, count: counts[c] }));
})();

// Vehicle count vs speed scatter (colored by class) for dashboard chart
export const SCATTER_POINTS = DATA.map(r => ({
  vehicles: r.Vehicle_Count,
  speed: r.Average_Speed,
  signal: r.Signal_Time,
  capacity: r.Road_Capacity,
  cls: r.Traffic_Level,
}));

export const DATASET_INFO = {
  name: "Traffic Density Dataset",
  source: "Generated per reference notebook (Smart_Traffic_Density_Prediction.ipynb)",
  features: FEATURES as unknown as string[],
  target: "Traffic_Level (Low / Medium / High)",
  records: DATA.length,
  trainSize: TRAIN.length,
  testSize: TEST.length,
  pipeline: "StandardScaler → LogisticRegression (multinomial, L2)",
};
