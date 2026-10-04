import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const STATIC = import.meta.env.VITE_STATIC === "1";
type Risk = "Low" | "Medium" | "High";
interface DiffOp { op: "equal" | "delete" | "insert" | "replace"; a: string; b: string }
interface Review { clause_type: string; rule_id: string | null; risk: Risk; explanation: string; suggested_text: string | null; diff?: DiffOp[]; failed?: boolean }
interface Clause { id: string; number: string; title: string; section: string; text: string; page: number; review: Review; applied: boolean }
interface Rule { id: string; title: string; standard: string; red_flags: string }
interface Doc {
  id: string; filename: string; status: "processing" | "ready" | "failed"; error?: string; pages?: number; perspective?: string;
  clauses?: Clause[]; scorecard?: { High: number; Medium: number; Low: number; overall: Risk };
  playbook?: { name: string; rules: Rule[] }; usage?: { usd: number; calls: number }; seconds?: number;
}

const token = () => sessionStorage.getItem("token") ?? "";
async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api/${path}`, { ...init, headers: { Authorization: `Bearer ${token()}`, ...(init.headers ?? {}) } });
  if (!res.ok) throw new Error(`${res.status}: ${JSON.stringify((await res.json().catch(() => ({}))).detail ?? res.statusText)}`);
  return (res.headers.get("content-type") ?? "").includes("json") ? res.json() : (res.text() as Promise<T>);
}

function Diff({ ops }: { ops: DiffOp[] }) {
  return (
    <div className="diff">
      <div><h3>Original</h3><p>{ops.map((o, i) => o.a && (o.op === "equal" ? <span key={i}>{o.a} </span> : <del key={i}>{o.a} </del>))}</p></div>
      <div><h3>AI proposal</h3><p>{ops.map((o, i) => o.b && (o.op === "equal" ? <span key={i}>{o.b} </span> : <ins key={i}>{o.b} </ins>))}</p></div>
    </div>
  );
}

export function App() {
  const [doc, setDoc] = useState<Doc | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [filter, setFilter] = useState<Risk | "all">("all");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<string[] | null>(null);       // clause ids from the last search, best first
  const [error, setError] = useState("");
  const [signedIn, setSignedIn] = useState(STATIC || !!token());
  const middle = useRef<HTMLDivElement>(null);

  const open = useCallback(async (id: string) => {
    try {
      for (;;) {                                                   // poll while the background analysis runs
        const d = await api<Doc>(`documents/${id}`);
        setDoc(d);
        if (d.status !== "processing") break;
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => {
    if (!signedIn) return;
    if (STATIC) { fetch("./data/document.json").then((r) => r.json()).then(setDoc); return; }
    api<Doc[]>("documents").then((list) => { if (list[0]) open(list[0].id); else setDoc({ id: "", filename: "", status: "ready" }); })
      .catch((e) => { setError(String(e)); if (String(e).startsWith("Error: 401")) { sessionStorage.removeItem("token"); setSignedIn(false); } });
  }, [signedIn, open]);

  const clauses = doc?.clauses ?? [];
  const current = clauses.find((c) => c.id === sel) ?? null;
  const rules = useMemo(() => Object.fromEntries((doc?.playbook?.rules ?? []).map((r) => [r.id, r])), [doc]);
  const listed = useMemo(() => {
    const base = clauses.filter((c) => filter === "all" || c.review.risk === filter);
    if (!hits) return base;
    return hits.map((id) => base.find((c) => c.id === id)).filter((c): c is Clause => !!c);
  }, [clauses, filter, hits]);

  const select = (id: string) => {
    setSel(id);
    middle.current?.querySelector(`#${id}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  const search = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return setHits(null);
    if (STATIC) {                                                  // demo: plain keyword match, no server to embed the query
      const words = query.toLowerCase().split(/\s+/).filter(Boolean);
      return setHits(clauses.map((c) => ({ c, n: words.filter((w) => `${c.title} ${c.text}`.toLowerCase().includes(w)).length }))
        .filter((x) => x.n > 0).sort((a, b) => b.n - a.n).slice(0, 12).map((x) => x.c.id));
    }
    try { setHits((await api<{ id: string }[]>(`documents/${doc!.id}/search?q=${encodeURIComponent(query)}`)).map((h) => h.id)); }
    catch (err) { setError(String(err)); }
  };
  const apply = async (c: Clause, applied: boolean) => {
    try {
      if (!STATIC) await api(`documents/${doc!.id}/clauses/${c.id}/apply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ applied }) });
      setDoc((d) => d && { ...d, clauses: d.clauses!.map((x) => (x.id === c.id ? { ...x, applied } : x)) });
    } catch (e) { setError(String(e)); }
  };
  const upload = async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    try { const r = await api<{ id: string }>("documents", { method: "POST", body }); setSel(null); setHits(null); await open(r.id); }
    catch (e) { setError(String(e)); }
  };
  const exportText = async () => {
    const text = STATIC
      ? clauses.map((c) => `${[c.number, c.title].filter(Boolean).join(" ")}\n${c.applied && c.review.suggested_text ? c.review.suggested_text : c.text}`).join("\n\n")
      : await api<string>(`documents/${doc!.id}/export`);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    a.download = `${doc!.filename.replace(/\.\w+$/, "")}-revised.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (!signedIn)
    return (
      <form className="signin" onSubmit={(e) => { e.preventDefault(); sessionStorage.setItem("token", String(new FormData(e.currentTarget).get("token"))); setSignedIn(true); }}>
        <h1>Contract &amp; RFP Workspace</h1>
        <label>API token<input name="token" type="password" required autoComplete="off" /></label>
        <button>Sign in</button>
        {error && <p className="error" role="alert">{error}</p>}
      </form>
    );

  const card = doc?.scorecard;
  const applied = clauses.filter((c) => c.applied).length;
  return (
    <div className="app">
      <header>
        <h1>Contract &amp; RFP Workspace</h1>
        {doc?.filename && <span className="file" title={doc.filename}>{doc.filename}{doc.pages ? ` · ${doc.pages} pages · ${clauses.length} clauses` : ""}</span>}
        {card && <span className="score">
          <b className="risk High">{card.High} high</b><b className="risk Medium">{card.Medium} medium</b><b className="risk Low">{card.Low} low</b>
        </span>}
        <span className="actions">
          {!STATIC && <label className="button">Upload<input type="file" hidden accept=".pdf,.docx,.png,.jpg,.jpeg,.tif,.tiff" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} /></label>}
          {doc?.status === "ready" && clauses.length > 0 && <button onClick={exportText}>Export text{applied ? ` (${applied} draft${applied === 1 ? "" : "s"} applied)` : ""}</button>}
        </span>
      </header>
      {STATIC && <p className="banner">Demo: a recorded analysis of a public 52-page SEC exhibit against an invented example playbook. Not legal advice; the reviews were not checked by a lawyer. Search here is keyword-only and changes are not saved.</p>}
      {error && <p className="error" role="alert">{error} <button onClick={() => setError("")}>Dismiss</button></p>}
      {!doc ? <p className="empty">Loading…</p>
        : doc.status === "processing" ? <p className="empty">Analysing {doc.filename}… parsing, chunking by clause, then reviewing each clause against the playbook.</p>
        : doc.status === "failed" ? <p className="error">Analysis failed: {doc.error}</p>
        : clauses.length === 0 ? <p className="empty">Upload a contract or RFP (PDF, DOCX or a scanned image) to start.</p>
        : (
          <div className="panes">
            <nav aria-label="Document contents">
              <form onSubmit={search} className="search">
                <input type="search" placeholder="Search the document" value={query} onChange={(e) => { setQuery(e.target.value); if (!e.target.value) setHits(null); }} aria-label="Search the document" />
              </form>
              <div className="seg" role="group" aria-label="Filter by risk">
                {(["all", "High", "Medium", "Low"] as const).map((f) => <button key={f} aria-pressed={filter === f} onClick={() => setFilter(f)}>{f}</button>)}
              </div>
              {hits && <p className="hint">{listed.length} result{listed.length === 1 ? "" : "s"}, best first</p>}
              <ol>
                {listed.map((c, i) => (
                  <li key={c.id}>
                    {!hits && c.section && c.section !== listed[i - 1]?.section && <h2>{c.section}</h2>}
                    <button className={sel === c.id ? "on" : ""} onClick={() => select(c.id)}>
                      <i className={`dot ${c.review.risk}`} aria-label={`${c.review.risk} risk`} />
                      <span>{c.number} {c.title}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </nav>
            <article ref={middle} aria-label="Document">
              {clauses.map((c) => (
                <section key={c.id} id={c.id} className={`clause ${c.review.risk} ${sel === c.id ? "on" : ""}`} onClick={() => setSel(c.id)}>
                  <h2>{c.number} {c.title} <small>p. {c.page}</small>
                    {c.review.risk !== "Low" && <b className={`risk ${c.review.risk}`}>{c.review.risk}</b>}
                    {c.applied && <b className="risk applied">AI draft applied</b>}</h2>
                  {(c.applied && c.review.suggested_text ? c.review.suggested_text : c.text).split("\n").map((p, i) => <p key={i}>{p}</p>)}
                </section>
              ))}
            </article>
            <aside aria-label="AI review">
              {!current ? <p className="empty">Select a clause to see its risk score, the reason and a suggested redraft.</p> : (
                <>
                  <h2>{current.number} {current.title}</h2>
                  <p><b className={`risk ${current.review.risk}`}>{current.review.risk} risk</b>
                    {current.review.rule_id && rules[current.review.rule_id] && <span className="rule"> Playbook: {rules[current.review.rule_id].title}</span>}</p>
                  <p>{current.review.explanation}</p>
                  {current.review.rule_id && rules[current.review.rule_id] && (
                    <details><summary>What the playbook says</summary>
                      <p><b>Standard.</b> {rules[current.review.rule_id].standard}</p>
                      <p><b>Red flags.</b> {rules[current.review.rule_id].red_flags}</p></details>
                  )}
                  {current.review.diff && <>
                    <Diff ops={current.review.diff} />
                    <button className="primary" onClick={() => apply(current, !current.applied)}>{current.applied ? "Undo AI draft" : "Apply AI draft"}</button>
                  </>}
                </>
              )}
            </aside>
          </div>
        )}
    </div>
  );
}
