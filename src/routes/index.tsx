import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  Activity, Gauge, TrendingUp, Database, Car, Zap, Target,
  AlertTriangle, CheckCircle2, Timer, Ruler,
} from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  BarChart, Bar, ScatterChart, Scatter, ZAxis, Legend, ReferenceLine,
} from "recharts";
import {
  predictTraffic, MODEL_METRICS, ROC_CURVES, FEATURE_IMPORTANCE,
  CLASS_DISTRIBUTION, SCATTER_POINTS, DATASET_INFO, CLASSES,
  type PredictionInput, type TrafficClass,
} from "@/lib/traffic-model";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Smart Traffic Density Prediction System" },
      { name: "description", content: "Multinomial logistic regression classifier for traffic density (Low / Medium / High) with ROC analysis, confusion matrix and dashboard reporting." },
      { property: "og:title", content: "Smart Traffic Density Prediction System" },
      { property: "og:description", content: "Predict traffic density with logistic regression, ROC curves, and a live dashboard." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dashboard,
});

const CLASS_COLORS: Record<TrafficClass, string> = {
  Low: "var(--success)",
  Medium: "var(--warning)",
  High: "var(--danger)",
};

function Dashboard() {
  const [input, setInput] = useState<PredictionInput>({
    Vehicle_Count: 300, Average_Speed: 35, Signal_Time: 60, Road_Capacity: 400,
  });

  const prediction = useMemo(() => predictTraffic(input), [input]);
  const topClass = prediction.label;
  const topP = prediction.topProb;

  const probData = prediction.probs.map(p => ({
    cls: p.cls, prob: +(p.p * 100).toFixed(1),
  }));

  const rocData = useMemo(() => {
    // merge per-class ROC into a single chart-friendly array of {fpr, Low, Medium, High}
    // sample 40 points per class
    const N = 40;
    const sample = (pts: { fpr: number; tpr: number }[]) => {
      if (pts.length <= N) return pts;
      const out: { fpr: number; tpr: number }[] = [];
      const step = pts.length / N;
      for (let i = 0; i < N; i++) out.push(pts[Math.floor(i * step)]);
      out.push(pts[pts.length - 1]);
      return out;
    };
    const perClass = ROC_CURVES.map(c => ({ cls: c.cls, pts: sample(c.points) }));
    // Build union of fprs
    const merged: Record<number, { fpr: number; Low?: number; Medium?: number; High?: number }> = {};
    for (const c of perClass) {
      for (const p of c.pts) {
        const key = +p.fpr.toFixed(3);
        if (!merged[key]) merged[key] = { fpr: key };
        (merged[key] as Record<string, number>)[c.cls] = p.tpr;
      }
    }
    return Object.values(merged).sort((a, b) => a.fpr - b.fpr);
  }, []);

  return (
    <div style={{ minHeight: "100vh" }}>
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "32px 24px 64px" }}>
        <Header />

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginTop: 28 }}>
          <StatCard icon={<Target size={20} />} label="Test Accuracy" value={`${(MODEL_METRICS.accuracy * 100).toFixed(1)}%`} accent="var(--primary)" />
          <StatCard icon={<Database size={20} />} label="Dataset Records" value={`${DATASET_INFO.records}`} accent="var(--accent)" />
          <StatCard icon={<Activity size={20} />} label="Train / Test" value={`${DATASET_INFO.trainSize} / ${DATASET_INFO.testSize}`} accent="var(--chart-4)" />
          <StatCard icon={<Zap size={20} />} label="Classes" value={CLASSES.join(" · ")} accent="var(--chart-5)" />
        </div>

        {/* Predictor */}
        <Section title="Live Prediction" icon={<Gauge size={18} />}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }} className="grid-2">
            <Card>
              <div style={{ display: "grid", gap: 14 }}>
                <Slider label="Vehicle Count" icon={<Car size={14} />} value={input.Vehicle_Count} min={50} max={500} step={1}
                  onChange={v => setInput({ ...input, Vehicle_Count: v })} unit="veh" />
                <Slider label="Average Speed" icon={<Zap size={14} />} value={input.Average_Speed} min={10} max={80} step={1}
                  onChange={v => setInput({ ...input, Average_Speed: v })} unit="km/h" />
                <Slider label="Signal Time" icon={<Timer size={14} />} value={input.Signal_Time} min={20} max={120} step={1}
                  onChange={v => setInput({ ...input, Signal_Time: v })} unit="s" />
                <Slider label="Road Capacity" icon={<Ruler size={14} />} value={input.Road_Capacity} min={100} max={600} step={1}
                  onChange={v => setInput({ ...input, Road_Capacity: v })} unit="veh/h" />
              </div>
            </Card>

            <Card>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, minHeight: 200 }}>
                <div style={{ fontSize: 13, color: "var(--muted-foreground)", textTransform: "uppercase", letterSpacing: 1.5 }}>
                  Predicted Traffic Level
                </div>
                <div style={{
                  fontSize: 56, fontWeight: 800, letterSpacing: -1.5,
                  color: CLASS_COLORS[topClass],
                }}>
                  {topClass}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--muted-foreground)" }}>
                  {topClass === "High" ? <AlertTriangle size={16} color="var(--danger)" /> : <CheckCircle2 size={16} color="var(--success)" />}
                  <span>Confidence: <b style={{ color: "var(--foreground)" }}>{(topP * 100).toFixed(1)}%</b></span>
                </div>
                <div style={{ width: "100%", height: 160, marginTop: 8 }}>
                  <ResponsiveContainer>
                    <BarChart data={probData} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" domain={[0, 100]} stroke="var(--muted-foreground)" fontSize={11} />
                      <YAxis type="category" dataKey="cls" stroke="var(--muted-foreground)" fontSize={12} />
                      <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v}%`} />
                      <Bar dataKey="prob" radius={[0, 6, 6, 0]} fill="var(--primary)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Card>
          </div>
        </Section>

        {/* Class distribution + scatter */}
        <Section title="Dataset Overview" icon={<Database size={18} />}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 20 }} className="grid-2">
            <Card>
              <CardTitle>Class Distribution</CardTitle>
              <div style={{ height: 260 }}>
                <ResponsiveContainer>
                  <BarChart data={CLASS_DISTRIBUTION}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="cls" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis stroke="var(--muted-foreground)" fontSize={12} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="count" fill="var(--primary)" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card>
              <CardTitle>Vehicle Count vs Average Speed (by class)</CardTitle>
              <div style={{ height: 260 }}>
                <ResponsiveContainer>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" dataKey="vehicles" name="Vehicles" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis type="number" dataKey="speed" name="Speed" stroke="var(--muted-foreground)" fontSize={12} />
                    <ZAxis range={[60, 60]} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3" }} />
                    <Legend />
                    {(["Low", "Medium", "High"] as TrafficClass[]).map(c => (
                      <Scatter key={c} name={c} data={SCATTER_POINTS.filter(p => p.cls === c)} fill={CLASS_COLORS[c]} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </Section>

        {/* ROC + Feature importance */}
        <Section title="ROC Analysis (One-vs-Rest)" icon={<TrendingUp size={18} />}>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }} className="grid-2">
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                <CardTitle>ROC Curve per Class</CardTitle>
                <div style={{ display: "flex", gap: 12, fontSize: 12 }}>
                  {ROC_CURVES.map(c => (
                    <span key={c.cls} style={{ color: CLASS_COLORS[c.cls] }}>
                      {c.cls} AUC = {c.auc.toFixed(3)}
                    </span>
                  ))}
                </div>
              </div>
              <div style={{ height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={rocData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="var(--muted-foreground)" fontSize={11}
                      label={{ value: "False Positive Rate", position: "insideBottom", offset: -2, fill: "var(--muted-foreground)", fontSize: 11 }} />
                    <YAxis type="number" domain={[0, 1]} stroke="var(--muted-foreground)" fontSize={11}
                      label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "var(--muted-foreground)", fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="Low" stroke={CLASS_COLORS.Low} strokeWidth={2} dot={false} connectNulls />
                    <Line type="monotone" dataKey="Medium" stroke={CLASS_COLORS.Medium} strokeWidth={2} dot={false} connectNulls />
                    <Line type="monotone" dataKey="High" stroke={CLASS_COLORS.High} strokeWidth={2} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card>
              <CardTitle>Feature Importance</CardTitle>
              <div style={{ height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" stroke="var(--muted-foreground)" fontSize={11} />
                    <YAxis type="category" dataKey="feature" width={110} stroke="var(--muted-foreground)" fontSize={12} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="weight" fill="var(--accent)" radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </Section>

        {/* Classification report + confusion matrix */}
        <Section title="Classification Report" icon={<Target size={18} />}>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }} className="grid-2">
            <Card>
              <CardTitle>Per-Class Metrics (Held-out Test Set)</CardTitle>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                  <thead>
                    <tr style={{ color: "var(--muted-foreground)", textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                      <th style={{ padding: "10px 8px" }}>Class</th>
                      <th style={{ padding: "10px 8px" }}>Precision</th>
                      <th style={{ padding: "10px 8px" }}>Recall</th>
                      <th style={{ padding: "10px 8px" }}>F1-Score</th>
                      <th style={{ padding: "10px 8px" }}>Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {MODEL_METRICS.perClass.map(row => (
                      <tr key={row.cls} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "10px 8px", color: CLASS_COLORS[row.cls], fontWeight: 700 }}>{row.cls}</td>
                        <td style={{ padding: "10px 8px" }}>{row.precision.toFixed(3)}</td>
                        <td style={{ padding: "10px 8px" }}>{row.recall.toFixed(3)}</td>
                        <td style={{ padding: "10px 8px" }}>{row.f1.toFixed(3)}</td>
                        <td style={{ padding: "10px 8px" }}>{row.support}</td>
                      </tr>
                    ))}
                    <tr>
                      <td style={{ padding: "10px 8px", fontWeight: 700 }}>Overall</td>
                      <td colSpan={3} style={{ padding: "10px 8px", color: "var(--muted-foreground)" }}>Accuracy</td>
                      <td style={{ padding: "10px 8px", fontWeight: 700, color: "var(--primary)" }}>{(MODEL_METRICS.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Card>
            <Card>
              <CardTitle>Confusion Matrix</CardTitle>
              <ConfusionMatrix cm={MODEL_METRICS.confusionMatrix} classes={CLASSES} />
            </Card>
          </div>
        </Section>

        <Section title="Dataset & Pipeline" icon={<Database size={18} />}>
          <Card>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, fontSize: 14 }}>
              <InfoRow k="Dataset" v={DATASET_INFO.name} />
              <InfoRow k="Source" v={DATASET_INFO.source} />
              <InfoRow k="Records" v={String(DATASET_INFO.records)} />
              <InfoRow k="Features" v={DATASET_INFO.features.join(", ")} />
              <InfoRow k="Target" v={DATASET_INFO.target} />
              <InfoRow k="Pipeline" v={DATASET_INFO.pipeline} />
            </div>
          </Card>
        </Section>

        <footer style={{ marginTop: 40, textAlign: "center", color: "var(--muted-foreground)", fontSize: 13 }}>
          Smart Traffic Density Prediction · Logistic Regression · Trained live in-browser
        </footer>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .grid-2 { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}

const tooltipStyle: React.CSSProperties = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

function Header() {
  return (
    <header>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 44, height: 44, borderRadius: 12,
          background: "linear-gradient(135deg, var(--primary), var(--accent))",
          display: "grid", placeItems: "center", color: "var(--primary-foreground)",
        }}>
          <Car size={22} />
        </div>
        <div>
          <h1 style={{ margin: 0, fontSize: 26, letterSpacing: -0.5 }}>Smart Traffic Density Prediction</h1>
          <p style={{ margin: 0, color: "var(--muted-foreground)", fontSize: 13 }}>
            Logistic Regression · Classification Metrics · ROC Analysis · Dashboard Reporting
          </p>
        </div>
      </div>
    </header>
  );
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: 32 }}>
      <h2 style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15, textTransform: "uppercase", letterSpacing: 1.5, color: "var(--muted-foreground)", margin: "0 0 12px", fontWeight: 600 }}>
        <span style={{ color: "var(--primary)" }}>{icon}</span>{title}
      </h2>
      {children}
    </section>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: "var(--card)", color: "var(--card-foreground)",
      border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
      padding: 18,
    }}>
      {children}
    </div>
  );
}

function CardTitle({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>{children}</div>;
}

function StatCard({ icon, label, value, accent }: { icon: React.ReactNode; label: string; value: string; accent: string }) {
  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)", padding: 16,
      display: "flex", alignItems: "center", gap: 14,
    }}>
      <div style={{
        width: 40, height: 40, borderRadius: 10, display: "grid", placeItems: "center",
        background: `color-mix(in oklch, ${accent} 20%, transparent)`, color: accent,
      }}>{icon}</div>
      <div>
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
        <div style={{ fontSize: 18, fontWeight: 700 }}>{value}</div>
      </div>
    </div>
  );
}

function Slider({ label, icon, value, min, max, step, onChange, unit }: {
  label: string; icon: React.ReactNode; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; unit: string;
}) {
  return (
    <label style={{ display: "block" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--muted-foreground)" }}>
          {icon}{label}
        </span>
        <span style={{ fontWeight: 700 }}>{value} <span style={{ color: "var(--muted-foreground)", fontWeight: 400 }}>{unit}</span></span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "var(--primary)" }} />
    </label>
  );
}

function ConfusionMatrix({ cm, classes }: { cm: number[][]; classes: TrafficClass[] }) {
  const max = Math.max(1, ...cm.flat());
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 13, margin: "0 auto" }}>
        <thead>
          <tr>
            <th></th>
            {classes.map(c => (
              <th key={c} style={{ padding: 6, color: "var(--muted-foreground)", fontWeight: 500 }}>Pred {c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <th style={{ padding: 6, color: "var(--muted-foreground)", fontWeight: 500, textAlign: "right" }}>Actual {classes[i]}</th>
              {row.map((v, j) => {
                const intensity = v / max;
                return (
                  <td key={j} style={{
                    width: 60, height: 46, textAlign: "center", fontWeight: 700,
                    background: `color-mix(in oklch, var(--primary) ${Math.round(intensity * 65)}%, transparent)`,
                    border: "1px solid var(--border)", borderRadius: 6,
                    color: intensity > 0.5 ? "var(--primary-foreground)" : "var(--foreground)",
                  }}>{v}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InfoRow({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, color: "var(--muted-foreground)" }}>{k}</div>
      <div style={{ marginTop: 2 }}>{v}</div>
    </div>
  );
}
