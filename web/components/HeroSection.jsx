
// Hero Section — Daylight Lab (light, paper-like) with IGV legend
const EXAMPLE_SGRNA = "GAGTCCGAGCAGAAGAAGAATGG";
const EXAMPLE_DNA   = "GAGTCCGAGCAGAAGAAGAATGG";

const FAMOUS_PAIRS = [
  { id:'cftr-on',   name:'CFTR on-target',     sg:'GAGTCCGAGCAGAAGAAGAATGG', dna:'GAGTCCGAGCAGAAGAAGAATGG', del:false, note:'Perfect match · expect HIGH cleavage' },
  { id:'cftr-off',  name:'CFTR seed off-tgt',  sg:'GAGTCCGAGCAGAAGAAGAATGG', dna:'GAGTCCGAGCATAATAATAATGG', del:false, note:'3 seed mismatches · expect LOW' },
  { id:'cftr-d508', name:'CFTR ΔF508',         sg:'TGGCACCATTAAAGAAAATATGG', dna:'TGGCACCATTAAAGAAAATATGG', del:true,  note:'Deletion-aware (toggle ON)' },
  { id:'emx1',      name:'EMX1 classic',       sg:'GAGTCCGAGCAGAAGAAGAAGGG', dna:'GAGTCCGAGCAGAAGAAGAAGGG', del:false, note:'EMX1 reference target' },
  { id:'distal',    name:'Distal mismatches',  sg:'GAGTCCGAGCAGAAGAAGAATGG', dna:'AAATCCGAGCAGAAGAAGAATGG', del:false, note:'PAM-distal mismatches · tolerated' },
];

const BASE_COLORS = { A:'#00C853', T:'#DC2626', G:'#B7791F', C:'#2962FF' };

const HeroSection = ({ onRun }) => {
  const [sgRNA, setSgRNA] = React.useState('');
  const [dna, setDna]     = React.useState('');
  const [hasDel, setHasDel] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [loadingStep, setLoadingStep] = React.useState(0);
  const [toast, setToast] = React.useState(null);
  const sgRef = React.useRef(null);

  const loadingSteps = ['Encoding sequences…', 'Running DNABERT-2…', 'CNN feature extraction…', 'KAN decision layer…', 'Computing cleavage probability…'];

  React.useEffect(() => {
    const handler = (e) => { if (e.key === '/') { e.preventDefault(); sgRef.current?.focus(); } };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const validateSgRNA = (s) => ({
    lengthOk: s.length === 23,
    pamOk: s.length >= 2 && s.slice(-2).toUpperCase() === 'GG',
  });
  const validateDNA = (s) => ({ lengthOk: s.length === 23 });
  const countMismatches = (a, b) => {
    let c = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i]?.toUpperCase() !== b[i]?.toUpperCase()) c++;
    return c;
  };

  const sgV = validateSgRNA(sgRNA);
  const dnV = validateDNA(dna);
  const mm = sgRNA && dna ? countMismatches(sgRNA, dna) : null;

  const API_URL = (window.NEURO_API_URL || 'http://localhost:8000') + '/api/predict';

  const handleRun = async () => {
    if (sgRNA.length !== 23 || dna.length !== 23) return;
    setLoading(true);
    setLoadingStep(0);
    const interval = setInterval(() => setLoadingStep(s => (s + 1) % loadingSteps.length), 600);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sgrna: sgRNA, dna: dna, has_deletion: hasDel }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const result = await res.json();
      clearInterval(interval);
      setLoading(false);

      setToast(`Prediction complete in ${result.elapsed ?? '?'}s ✓`);
      setTimeout(() => setToast(null), 3500);
      onRun(result);

      setTimeout(() => {
        const el = document.getElementById('scan-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 200);
    } catch (e) {
      clearInterval(interval);
      setLoading(false);
      setToast(`⚠ Backend error: ${e.message}. Is uvicorn running on :8000?`);
      setTimeout(() => setToast(null), 6000);
    }
  };

  const renderColoredSeq = (s) =>
    s.split('').map((b, i) => (
      <span key={i} style={{ color: BASE_COLORS[b.toUpperCase()] || 'var(--ink)' }}>{b}</span>
    ));

  return (
    <section style={heroSectionStyles}>
      {/* Top eyebrow */}
      <div style={pillStyles}>
        <span style={{color:'var(--margin-red)', fontWeight:700}}>§</span>
        <span>IEEE ICAUC 2026 · DNABERT-2 + LoRA + KAN</span>
      </div>

      {/* Headline */}
      <h1 style={h1Styles}>
        Neuro-CRISPR<br />
        <span style={{ color: 'var(--teal)', fontStyle: 'italic', fontWeight: 600 }}>
          Off-Target Cleavage Prediction
        </span>
      </h1>
      <p style={subheadStyles}>
        Predicts <b style={{color:'var(--ink)'}}>Cas9 cleavage probability</b> for any sgRNA + DNA pair.<br/>
        AUROC <b>0.873</b> on CFTR · sub-second inference on a single GPU.
      </p>

      {/* IGV base-color legend */}
      <div style={legendRowStyles}>
        <span style={{...legendLabel}}>BASE COLORS · IGV/UCSC</span>
        {Object.entries(BASE_COLORS).map(([b, c]) => (
          <span key={b} style={legendChip}>
            <span style={{ ...legendDot, background: c }} />
            <span style={{ fontFamily:"'JetBrains Mono', monospace", color: c, fontWeight: 700 }}>{b}</span>
          </span>
        ))}
      </div>

      {/* Famous examples chip row */}
      <div style={{display:'flex', flexWrap:'wrap', gap:8, justifyContent:'center', marginBottom:18, maxWidth:760}}>
        <span style={presetLabel}>PRESETS</span>
        {FAMOUS_PAIRS.map(p => (
          <button key={p.id}
            title={p.note}
            onClick={() => { setSgRNA(p.sg); setDna(p.dna); setHasDel(p.del); }}
            style={presetChip}
            onMouseEnter={e => { e.currentTarget.style.borderColor='var(--teal)'; e.currentTarget.style.background='var(--teal-tint)'; e.currentTarget.style.color='var(--teal)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor='var(--hairline)'; e.currentTarget.style.background='var(--paper-2)'; e.currentTarget.style.color='var(--ink-soft)'; }}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* Paper input card */}
      <div className="hover-lift" style={paperCardStyles}>
        <div style={{display:'flex', gap:18, flexWrap:'wrap'}}>
          {/* sgRNA input */}
          <div style={{flex:1, minWidth:240}}>
            <div style={inputLabelRow}>
              <span style={labelStyles}>sgRNA <span style={muted}>· 23 nt</span></span>
              <button style={loadExStyles} onClick={() => setSgRNA(EXAMPLE_SGRNA)}>load example</button>
            </div>
            <input
              ref={sgRef}
              value={sgRNA}
              onChange={e => setSgRNA(e.target.value.toUpperCase().replace(/[^ATGC]/g,'').slice(0,23))}
              placeholder="ATCGATCGATCGATCGATCGAGG"
              style={{...monoInputStyles, borderColor: sgRNA && !sgV.lengthOk ? 'var(--danger)' : 'var(--hairline)'}}
              maxLength={23}
              spellCheck={false}
            />
            {sgRNA && (
              <div style={coloredSeqStrip}>{renderColoredSeq(sgRNA)}</div>
            )}
            <div style={chipRowStyles}>
              {sgRNA.length > 0 && (
                <span style={{...chipStyles, ...(sgV.lengthOk ? chipSafe : chipDanger)}}>
                  {sgV.lengthOk ? '✓' : '✗'} {sgRNA.length}/23 nt
                </span>
              )}
              {sgRNA.length === 23 && (
                <span style={{...chipStyles, ...(sgV.pamOk ? chipGold : chipDanger)}}>
                  {sgV.pamOk ? '✓ PAM NGG intact' : '! check PAM (NGG)'}
                </span>
              )}
            </div>
          </div>

          {/* DNA input */}
          <div style={{flex:1, minWidth:240}}>
            <div style={inputLabelRow}>
              <span style={labelStyles}>DNA target <span style={muted}>· 23 nt</span></span>
              <button style={loadExStyles} onClick={() => setDna(EXAMPLE_DNA)}>load example</button>
            </div>
            <input
              value={dna}
              onChange={e => setDna(e.target.value.toUpperCase().replace(/[^ATGC]/g,'').slice(0,23))}
              placeholder="ATCGATCGATCGATCGATCGAGG"
              style={{...monoInputStyles, borderColor: dna && !dnV.lengthOk ? 'var(--danger)' : 'var(--hairline)'}}
              maxLength={23}
              spellCheck={false}
            />
            {dna && (
              <div style={coloredSeqStrip}>{renderColoredSeq(dna)}</div>
            )}
            <div style={chipRowStyles}>
              {dna.length > 0 && (
                <span style={{...chipStyles, ...(dnV.lengthOk ? chipSafe : chipDanger)}}>
                  {dnV.lengthOk ? '✓' : '✗'} {dna.length}/23 nt
                </span>
              )}
              {sgRNA && dna && sgRNA.length === 23 && dna.length === 23 && mm !== null && (
                <span style={{...chipStyles, ...(mm === 0 ? chipSafe : mm <= 2 ? chipGold : chipDanger)}}>
                  {mm} mismatch{mm !== 1 ? 'es' : ''}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Deletion toggle */}
        <div style={{display:'flex', alignItems:'center', gap:10, marginTop:18}}>
          <div
            onClick={() => setHasDel(!hasDel)}
            style={{
              width: 42, height: 22, borderRadius: 11,
              background: hasDel ? 'var(--teal)' : 'var(--paper-3)',
              position: 'relative', cursor: 'pointer', transition: 'background 0.2s',
              border: '1px solid var(--hairline)',
            }}
          >
            <div style={{
              position: 'absolute', top: 2, left: hasDel ? 22 : 2,
              width: 16, height: 16, borderRadius: '50%',
              background: '#fff', boxShadow:'0 1px 2px rgba(0,0,0,0.2)',
              transition: 'left 0.2s',
            }} />
          </div>
          <span style={{color:'var(--ink-soft)', fontSize:13}}>
            Has deletion (e.g. ΔF508)
          </span>
          <span title="Null Tensor 5-channel encoding adds an explicit GAP channel for deletion awareness" style={{width:16,height:16,borderRadius:'50%',background:'var(--paper-3)',color:'var(--ink-faint)',display:'inline-flex',alignItems:'center',justifyContent:'center',fontSize:10,cursor:'help',flexShrink:0,fontWeight:700}}>?</span>
        </div>

        {/* CTA button */}
        <button
          onClick={handleRun}
          disabled={loading || sgRNA.length !== 23 || dna.length !== 23}
          style={{
            ...ctaStyles,
            opacity: loading || sgRNA.length !== 23 || dna.length !== 23 ? 0.5 : 1,
            cursor: loading || sgRNA.length !== 23 || dna.length !== 23 ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? (
            <span style={{display:'flex',alignItems:'center',gap:10,justifyContent:'center'}}>
              <span style={{animation:'spin 0.8s linear infinite', display:'inline-block', fontSize:16}}>⟳</span>
              {loadingSteps[loadingStep]}
            </span>
          ) : 'Run prediction →'}
        </button>

        {/* Stat cards row */}
        <div style={{display:'flex', gap:10, marginTop:20, justifyContent:'center', flexWrap:'wrap'}}>
          {[['117M', 'params'], ['0.4s', 'inference'], ['95%', 'recall'], ['0.873','AUROC']].map(([val, lbl]) => (
            <div key={lbl} style={miniStatStyles}>
              <span style={{color:'var(--teal)', fontWeight:700, fontSize:18, fontFamily:"'Source Serif 4', serif"}}>{val}</span>
              <span style={{color:'var(--ink-faint)', fontSize:11, marginTop:2, letterSpacing:'0.06em', textTransform:'uppercase', fontWeight:600}}>{lbl}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Scroll hint */}
      <div style={{marginTop:48, display:'flex', flexDirection:'column', alignItems:'center', gap:6, opacity:0.6}}>
        <span style={{fontSize:11, letterSpacing:'0.18em', color:'var(--ink-faint)', fontWeight:600}}>SCROLL FOR ANALYSIS</span>
        <div style={{animation:'bounce 1.6s ease-in-out infinite', fontSize:18, color:'var(--teal)'}}>⌄</div>
      </div>

      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', top: 24, right: 24, zIndex: 9999,
          background: 'var(--paper)', border: '1px solid var(--teal)',
          padding: '12px 20px', borderRadius: 10,
          color: 'var(--teal)', fontWeight: 600, fontSize: 14,
          boxShadow: '0 12px 32px rgba(20,20,20,0.12)',
          animation: 'slideInRight 0.3s ease',
        }}>
          {toast}
        </div>
      )}
    </section>
  );
};

// ── Styles ─────────────────────────────────────────────
const heroSectionStyles = {
  position: 'relative', minHeight: '100vh', display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center', padding: '64px 24px 80px',
  zIndex: 1, textAlign: 'center',
};
const pillStyles = {
  display: 'inline-flex', alignItems: 'center', gap: 8,
  background: 'var(--paper-2)', border: '1px solid var(--hairline)',
  borderRadius: 999, padding: '5px 14px', fontSize: 12, color: 'var(--ink-soft)',
  marginBottom: 24, letterSpacing: '0.04em', fontWeight: 500,
  fontFamily: "'Source Serif 4', serif", fontStyle: 'italic',
};
const h1Styles = {
  fontSize: 'clamp(40px, 5.5vw, 68px)', fontWeight: 700, lineHeight: 1.06,
  letterSpacing: '-0.02em', color: 'var(--ink)', marginBottom: 18,
  fontFamily: "'Source Serif 4', serif",
};
const subheadStyles = {
  color: 'var(--ink-soft)', fontSize: 17, lineHeight: 1.65, marginBottom: 28,
  maxWidth: 580, fontFamily: "'Source Serif 4', serif",
};
const legendRowStyles = {
  display:'flex', gap:10, alignItems:'center', flexWrap:'wrap', justifyContent:'center',
  marginBottom: 18,
};
const legendLabel = {
  fontSize:10, letterSpacing:'0.18em', color:'var(--ink-faint)', fontWeight:700,
  marginRight: 4,
};
const legendChip = {
  display:'inline-flex', alignItems:'center', gap:6,
  background:'var(--paper)', border:'1px solid var(--hairline)',
  borderRadius:6, padding:'3px 8px', fontSize:11,
};
const legendDot = { width:8, height:8, borderRadius:2, display:'inline-block' };
const presetLabel = {
  fontSize:10, letterSpacing:'0.18em', color:'var(--ink-faint)', fontWeight:700,
  alignSelf:'center', marginRight:4,
};
const presetChip = {
  background:'var(--paper-2)',
  border:'1px solid var(--hairline)',
  borderRadius:6, padding:'5px 12px',
  color:'var(--ink-soft)',
  fontFamily:"'Inter',sans-serif",
  fontSize:12, fontWeight:500, letterSpacing:'0.01em',
  cursor:'pointer', display:'inline-flex', alignItems:'center', gap:6,
  transition:'all 0.15s',
};
const paperCardStyles = {
  width: '100%', maxWidth: 720,
  background: '#FFFDF7',
  border: '1px solid var(--hairline)',
  borderRadius: 14, padding: 30,
  boxShadow: '0 1px 0 rgba(20,20,20,0.02), 0 14px 40px rgba(20,20,20,0.06)',
  textAlign: 'left',
};
const inputLabelRow = { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom: 6 };
const labelStyles = {
  color:'var(--ink)', fontSize:12, fontWeight:600, letterSpacing:'0.08em',
  textTransform:'uppercase', fontFamily:"'Inter', sans-serif",
};
const muted = { color:'var(--ink-faint)', fontWeight:500, letterSpacing:'0.04em' };
const loadExStyles = {
  background:'none', border:'none', color:'var(--teal)', fontSize:11,
  cursor:'pointer', padding:0, fontFamily:"'Source Serif 4', serif",
  fontStyle:'italic',
};
const monoInputStyles = {
  width: '100%', background: '#fff',
  border: '1px solid var(--hairline)',
  borderRadius: 8, padding: '12px 14px', color: 'var(--ink)',
  fontFamily: "'JetBrains Mono', monospace", fontSize: 14, letterSpacing: '0.1em',
  outline: 'none', transition: 'border-color 0.2s, box-shadow 0.2s', boxSizing:'border-box',
};
const coloredSeqStrip = {
  marginTop: 6, padding: '6px 10px',
  background: 'var(--paper-2)',
  borderRadius: 6,
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 13, letterSpacing: '0.18em',
  fontWeight: 700,
  overflowX: 'auto', whiteSpace: 'nowrap',
};
const chipRowStyles = { display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' };
const chipStyles = {
  display:'inline-flex', alignItems:'center', borderRadius:4,
  padding:'2px 8px', fontSize:11, fontWeight:600, letterSpacing:'0.04em',
  fontFamily: "'Inter', sans-serif",
};
const chipSafe   = { background:'var(--safe-soft)',   color:'var(--safe)',   border:'1px solid rgba(22,101,52,0.2)' };
const chipDanger = { background:'var(--danger-soft)', color:'var(--danger)', border:'1px solid rgba(201,42,42,0.2)' };
const chipGold   = { background:'var(--rust-soft)',   color:'var(--rust)',   border:'1px solid rgba(180,83,9,0.2)' };
const ctaStyles = {
  width:'100%', marginTop:22, padding:'14px 24px', borderRadius:10,
  background:'var(--teal)',
  border:'none', color:'#fff',
  fontFamily:"'Inter', sans-serif",
  fontWeight:600, fontSize:15, letterSpacing:'0.02em', cursor:'pointer',
  transition:'all 0.2s', boxShadow:'0 6px 18px rgba(13,110,100,0.20)',
};
const miniStatStyles = {
  display:'flex', flexDirection:'column', alignItems:'center',
  background:'var(--paper-2)', border:'1px solid var(--hairline)',
  borderRadius:8, padding:'10px 18px', minWidth:90,
};

Object.assign(window, { HeroSection });
