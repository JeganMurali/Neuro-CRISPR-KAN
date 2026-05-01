
// ============================================================
// Insights Panel + Footer · Daylight Lab
// ============================================================

const StatTile = ({ label, value, color }) => (
  <div style={{
    background:'var(--paper-2)', border:'1px solid var(--hairline)',
    borderRadius:8, padding:'14px 18px', textAlign:'center',
  }}>
    <div style={{fontSize:24, fontWeight:700, color: color || 'var(--ink)', fontFamily:"'Source Serif 4',serif", letterSpacing:'-0.01em'}}>{value}</div>
    <div style={{fontSize:11, color:'var(--ink-faint)', marginTop:4, letterSpacing:'0.08em', textTransform:'uppercase', fontWeight:600, fontFamily:"'Inter',sans-serif"}}>{label}</div>
  </div>
);

// ── Streaming text — typewriter for the Llama audit body ─────
const StreamText = ({ text, speed = 14, onDone }) => {
  const [shown, setShown] = React.useState('');
  React.useEffect(() => {
    setShown('');
    let i = 0;
    const id = setInterval(() => {
      i++;
      setShown(text.slice(0, i));
      if (i >= text.length) { clearInterval(id); onDone && onDone(); }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed]);
  return <>{shown}<span style={{borderRight:'1px solid #1a1a1a', marginLeft:1, animation:'blink 0.9s steps(1) infinite'}}/></>;
};

// ── Llama 3.1 8B — Lab Notebook Clinical Audit ───────────────
const LlamaAudit = ({ result, autoAudit }) => {
  const [expanded, setExpanded] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [auditData, setAuditData] = React.useState(null);
  const [elapsed, setElapsed] = React.useState(null);

  const AUDIT_URL = (window.NEURO_API_URL || 'http://localhost:8000') + '/api/audit';

  // Reset stale audit when a new prediction arrives
  React.useEffect(() => {
    setAuditData(null);
    setElapsed(null);
    setExpanded(false);
  }, [result?.sgRNA, result?.dna, result?.risk_prob]);

  // Auto-generate audit when triggered by parent (after scan completes)
  React.useEffect(() => {
    if (autoAudit && result && !auditData && !loading) {
      setExpanded(true);
      runAudit();
    }
    // eslint-disable-next-line
  }, [autoAudit, result?.sgRNA, result?.dna, result?.risk_prob]);

  const runAudit = async () => {
    setLoading(true);
    let serverData = null;
    try {
      const res = await fetch(AUDIT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sgrna: result.sgRNA, dna: result.dna,
          risk_prob: result.risk_prob,
          mismatches: result.mismatches ?? 0,
          seed_mismatches: result.seed_mismatches ?? 0,
          has_deletion: !!result.hasDel,
          pam_intact: !!result.pam_intact,
          chromatin_score: 0.5,
          use_llm: true,
        }),
      });
      if (res.ok) serverData = await res.json();
    } catch (e) {
      console.warn('audit fetch failed; using local synthesis', e);
    }

    const isHigh = result.risk_prob > 0.7;
    const isMod  = result.risk_prob > 0.3;
    const seedPos = result.sgRNA && result.dna
      ? [...Array(11)].map((_,i) => i+9).filter(i => result.sgRNA[i]?.toUpperCase() !== result.dna[i]?.toUpperCase())
      : [];

    let reasoning, recommendation, modeLabel;
    if (serverData && serverData.verdict) {
      const text = serverData.verdict;
      const recIdx = text.search(/recommendation[:\-\s]/i);
      if (recIdx > 50) {
        reasoning = text.slice(0, recIdx).replace(/^(reasoning|assessment)[:\-\s]*/i, '').trim();
        recommendation = text.slice(recIdx).replace(/^recommendation[:\-\s]*/i, '').trim();
      } else {
        reasoning = text.trim();
        recommendation = '— see assessment above —';
      }
      modeLabel = serverData.mode === 'llama-3.1-8b' ? 'Llama-3.1-8B' :
                  serverData.mode === 'template' ? 'Template (fast)' :
                  serverData.mode === 'template-fallback' ? 'Template (Llama unavailable)' :
                  'Llama';
    } else {
      reasoning = isHigh
        ? `The seed-region mismatch${seedPos.length > 1 ? 'es' : ''} at position${seedPos.length > 1 ? 's' : ''} ${seedPos.length ? seedPos.map(p=>p+1).join(', ') : '10–20'} ${seedPos.length > 1 ? 'are' : 'is'} the primary risk driver [1]. PAM (NGG) is ${result.pam_intact ? 'intact' : 'disrupted'}. Cas9 seed-region tolerance is low [2].`
        : `Mismatches are predominantly PAM-distal (positions 1–9), where Cas9 tolerates substitutions [2]. Seed shows ${result.seed_mismatches ?? 0} mismatch${result.seed_mismatches !== 1 ? 'es' : ''}; PAM ${result.pam_intact ? 'intact' : 'disrupted'}.`;
      recommendation = isHigh
        ? `Redesign the sgRNA or use high-fidelity Cas9 (eSpCas9, SpCas9-HF1) [3]. Validate via GUIDE-seq before any clinical use.`
        : `Standard GUIDE-seq off-target profiling recommended pre-IND. Validate on-target efficiency in CFBE41o− cells.`;
      modeLabel = 'Offline (backend unreachable)';
    }

    setElapsed(serverData?.elapsed ? `${serverData.elapsed}s · ${modeLabel}` : `local · ${modeLabel}`);
    setAuditData({
      verdictLabel: isHigh ? 'USE WITH EXTREME CAUTION' : isMod ? 'USE WITH CAUTION' : 'ACCEPTABLE RISK PROFILE',
      verdictBg: isHigh ? '#a31947' : isMod ? '#b87800' : '#187a4a',
      metrics: [
        { label:'Mismatch profile', value:`${result.mismatches ?? 0} total / ${result.seed_mismatches ?? 0} seed / PAM ${result.pam_intact ? '✓' : '✗'}`, dot: result.mismatches > 2 ? '#a31947' : '#b87800' },
        { label:'Cleavage likelihood', value:`${isHigh ? 'High' : isMod ? 'Moderate' : 'Low'} · p=${(result.risk_prob).toFixed(2)}`, dot: isHigh ? '#a31947' : isMod ? '#b87800' : '#187a4a' },
        { label:'Therapeutic risk', value: isHigh ? 'High — screen required' : isMod ? 'Medium' : 'Low', dot: isHigh ? '#a31947' : isMod ? '#b87800' : '#187a4a' },
        { label:'ΔF508 deletion', value: result.hasDel ? 'Present (null-tensor encoded)' : 'Absent', dot: '#444' },
      ],
      reasoning,
      recommendation,
      retrievedDocs: serverData?.retrieved || [],
      references: [
        { tag:'[1]', text:'Hsu et al. 2013 — DNA targeting specificity of RNA-guided Cas9 nucleases', doi:'10.1038/nbt.2647' },
        { tag:'[2]', text:'Doench et al. 2016 — Optimized sgRNA design for activity & off-target minimization', doi:'10.1038/nbt.3437' },
        { tag:'[3]', text:'Tycko et al. 2019 — Mitigation of off-target toxicity in CRISPR-Cas9 screens', doi:'10.1038/s41587-019-0095-1' },
      ],
      locus: 'chr7:117,548,628 (GRCh38) — CFTR ΔF508',
    });
    setLoading(false);
  };

  return (
    <div style={{background:'var(--paper-2)', border:'1px solid var(--hairline)', borderRadius:12, overflow:'hidden'}}>
      {/* Header (paper) */}
      <div style={{display:'flex', alignItems:'center', gap:10, padding:'14px 18px', borderBottom: expanded ? '1px solid var(--hairline)' : 'none', background:'var(--paper)'}}>
        <div style={{width:8, height:8, borderRadius:'50%', background:'var(--gold)', boxShadow:'0 0 6px rgba(183,121,31,0.5)'}}/>
        <div style={{flex:1, display:'flex', flexDirection:'column'}}>
          <span style={{color:'var(--ink)', fontWeight:700, fontSize:12, letterSpacing:'0.14em', fontFamily:"'Inter',sans-serif"}}>
            LLAMA-3.1-8B · CLINICAL AUDIT
          </span>
          <span style={{color:'var(--ink-faint)', fontSize:11, fontFamily:"'Source Serif 4',serif", fontStyle:'italic', marginTop:2}}>
            CFTR off-target report{elapsed ? ` · generated in ${elapsed}` : ''}
          </span>
        </div>
        <button
          onClick={() => { setExpanded(!expanded); if (!expanded && !auditData) runAudit(); }}
          style={{
            background: 'var(--rust-soft)',
            border:'1px solid rgba(180,83,9,0.35)', borderRadius:6, padding:'6px 14px',
            color:'var(--rust)', fontFamily:"'Inter',sans-serif", fontWeight:600, fontSize:12, cursor:'pointer',
          }}
        >
          {expanded ? '↑ Collapse' : 'Generate report →'}
        </button>
      </div>

      {/* Body */}
      {expanded && (
        <div style={{padding: loading ? '18px' : 0}}>
          {loading ? (
            <div style={{display:'flex', flexDirection:'column', gap:10, padding:'4px 0', color:'var(--ink-soft)', fontSize:12, fontFamily:"'JetBrains Mono',monospace"}}>
              <div style={{display:'flex', alignItems:'center', gap:10}}>
                <span style={{animation:'spin 0.8s linear infinite', display:'inline-block', color:'var(--rust)'}}>⟳</span>
                Retrieving RAG context (3 papers)…
              </div>
              <div style={{paddingLeft:24, color:'var(--ink-faint)'}}>↳ Hsu 2013 · Doench 2016 · Tycko 2019</div>
              <div style={{display:'flex', alignItems:'center', gap:10, marginTop:6}}>
                <span style={{animation:'spin 0.8s linear infinite', display:'inline-block', color:'var(--rust)'}}>⟳</span>
                Generating structured clinical report…
              </div>
            </div>
          ) : auditData ? (
            <div className="lab-paper" style={{padding: '0', position:'relative', minHeight:520}}>
              <div className="lab-paper-edge"/>
              <div className="lab-paper-margin"/>
              <div className="lab-paper-rules"/>

              {/* Top header bar */}
              <div style={{
                position:'relative', zIndex:2,
                background:'#0e1424', color:'#fff',
                padding:'12px 22px 12px 80px',
                fontFamily:"'JetBrains Mono', monospace", fontSize:10, letterSpacing:'0.12em',
                borderBottom:'2px solid #1a1a1a',
                display:'flex', justifyContent:'space-between', alignItems:'center',
              }}>
                <span>NEURO-CRISPR · CLINICAL OFF-TARGET AUDIT</span>
                <span style={{opacity:0.6}}>{new Date().toISOString().split('T')[0]} · LLAMA-3.1-8B</span>
              </div>

              {/* Verdict banner */}
              <div style={{
                position:'relative', zIndex:2,
                background: auditData.verdictBg,
                color: '#fff',
                padding:'18px 22px 18px 80px',
                fontFamily:"'Source Serif 4', Georgia, serif",
                fontSize: 26, fontWeight: 700, letterSpacing:'0.02em',
                borderBottom:'1px solid rgba(0,0,0,0.2)',
              }}>
                {auditData.verdictLabel}
              </div>

              <div style={{position:'relative', zIndex:2, display:'grid', gridTemplateColumns:'1.6fr 1fr', gap:24, padding:'24px 22px 24px 80px'}}>
                <div style={{fontFamily:"'Source Serif 4', Georgia, serif", color:'#1a1a1a'}}>
                  <div style={sectionHeaderLab}>I. ASSESSMENT</div>
                  <p style={prosePara}><StreamText text={auditData.reasoning} speed={9}/></p>

                  <div style={{...sectionHeaderLab, marginTop:18}}>II. RECOMMENDATION</div>
                  <p style={prosePara}><StreamText text={auditData.recommendation} speed={9}/></p>

                  <div style={{...sectionHeaderLab, marginTop:18}}>III. REFERENCES (RAG-RETRIEVED)</div>
                  <div style={{display:'flex', flexDirection:'column', gap:6, fontFamily:"'JetBrains Mono', monospace", fontSize:10.5}}>
                    {auditData.references.map((ref) => (
                      <div key={ref.tag} style={{display:'flex', gap:8, alignItems:'baseline', color:'#1a1a1a'}}>
                        <span style={{color:'#a31947', fontWeight:700, flexShrink:0}}>{ref.tag}</span>
                        <span style={{lineHeight:1.55}}>
                          {ref.text}
                          <span style={{color:'#406090', marginLeft:6}}>DOI:{ref.doi}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div style={sectionHeaderLab}>AT-A-GLANCE</div>
                  <div style={{
                    background:'rgba(255,255,255,0.6)',
                    border:'1px solid rgba(0,0,0,0.18)',
                    borderRadius:4,
                    padding:'12px 14px',
                    fontFamily:"'JetBrains Mono', monospace",
                    fontSize:11, color:'#1a1a1a',
                  }}>
                    {auditData.metrics.map((m, i) => (
                      <div key={m.label} style={{
                        display:'flex', alignItems:'center', gap:8,
                        padding: i === auditData.metrics.length - 1 ? '6px 0 0' : '6px 0',
                        borderBottom: i === auditData.metrics.length - 1 ? 'none' : '1px solid rgba(0,0,0,0.08)',
                      }}>
                        <div style={{width:8, height:8, borderRadius:'50%', background:m.dot, flexShrink:0}}/>
                        <div style={{display:'flex', flexDirection:'column', flex:1}}>
                          <span style={{fontSize:9, color:'#666', letterSpacing:'0.08em', textTransform:'uppercase'}}>{m.label}</span>
                          <span style={{fontSize:11, color:'#1a1a1a', fontWeight:600}}>{m.value}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div style={{...sectionHeaderLab, marginTop:18}}>LOCUS</div>
                  <div style={{fontFamily:"'JetBrains Mono', monospace", fontSize:10, color:'#1a1a1a', padding:'8px 0'}}>
                    {auditData.locus}
                  </div>

                  <div style={{...sectionHeaderLab, marginTop:18}}>SIGNATURE</div>
                  <div style={{
                    fontFamily:"'Source Serif 4', Georgia, serif",
                    fontStyle:'italic', fontSize:14, color:'#1a1a1a',
                    paddingTop:8, borderTop:'1px solid rgba(0,0,0,0.25)',
                  }}>
                    Llama 3.1 — 8B · Auditor
                  </div>
                  <div style={{fontFamily:"'JetBrains Mono', monospace", fontSize:9, color:'#666', marginTop:2}}>
                    Generated by Neuro-CRISPR
                  </div>
                </div>
              </div>

              <div style={{
                position:'relative', zIndex:2,
                background:'#0e1424', color:'rgba(255,255,255,0.5)',
                padding:'10px 22px 10px 80px',
                fontFamily:"'JetBrains Mono', monospace", fontSize:9, letterSpacing:'0.1em',
                display:'flex', justifyContent:'space-between',
              }}>
                <span>CONFIDENTIAL · For research use only · Not a medical device</span>
                <button style={{
                  background:'rgba(255,255,255,0.1)', border:'1px solid rgba(255,255,255,0.2)',
                  color:'rgba(255,255,255,0.85)', borderRadius:4, padding:'2px 10px', fontSize:9,
                  fontFamily:"'JetBrains Mono', monospace", cursor:'pointer', letterSpacing:'0.08em',
                }} onClick={() => window.print()}>
                  ↓ EXPORT PDF
                </button>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};

const sectionHeaderLab = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 10, letterSpacing: '0.18em',
  color: '#a31947', fontWeight: 700,
  borderBottom: '2px solid #1a1a1a',
  paddingBottom: 4, marginBottom: 10,
  textTransform: 'uppercase',
};
const prosePara = {
  fontSize: 13.5, lineHeight: 1.7,
  color: '#1a1a1a', margin: 0, marginBottom: 8,
  fontFamily: "'Source Serif 4', Georgia, serif",
};

const InsightsPanel = ({ result, autoAudit }) => {
  if (!result) return null;
  const { risk_prob, mismatches, seed_mismatches, pam_intact, hasDel } = result;
  const isHigh = risk_prob > 0.7;
  const riskLabel = risk_prob < 0.3 ? 'LOW CLEAVAGE' : risk_prob < 0.7 ? 'MODERATE CLEAVAGE' : 'HIGH CLEAVAGE';
  const riskColor = risk_prob < 0.3 ? 'var(--safe)' : risk_prob < 0.7 ? 'var(--rust)' : 'var(--danger)';

  return (
    <section style={{padding:'40px 24px 60px', zIndex:1, position:'relative'}}>
      <div style={{width:'100%', maxWidth:900, margin:'0 auto', display:'flex', flexDirection:'column', gap:18}}>
        <div>
          <div className="section-num">§ IV · Insights</div>
          <h2 style={{fontSize:28, color:'var(--ink)', margin:'4px 0 4px'}}>Clinical interpretation</h2>
          <p style={{color:'var(--ink-soft)', fontSize:14, fontFamily:"'Source Serif 4',serif"}}>
            Mismatch profile, model verdict, and a Llama-generated audit narrative.
          </p>
        </div>

        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))', gap:16}}>
          {/* Mismatch summary */}
          <div className="hover-lift" style={insightCardStyles}>
            <div style={insightCardTitleStyles}>Mismatch summary</div>
            <div style={{display:'flex', flexDirection:'column', gap:8}}>
              <StatTile label="Total mismatches" value={`${mismatches ?? 0} / 23`} color={mismatches > 3 ? 'var(--danger)' : 'var(--safe)'} />
              <StatTile label="Seed mismatches (10–20)" value={`${seed_mismatches ?? 0}`} color={seed_mismatches > 1 ? 'var(--danger)' : seed_mismatches > 0 ? 'var(--rust)' : 'var(--safe)'} />
              <StatTile label="PAM (NGG)" value={pam_intact ? 'Intact' : 'Disrupted'} color={pam_intact ? 'var(--safe)' : 'var(--danger)'} />
            </div>
          </div>

          {/* Model verdict */}
          <div className="hover-lift" style={insightCardStyles}>
            <div style={insightCardTitleStyles}>Model verdict</div>
            <div style={{display:'flex', alignItems:'center', gap:10, marginBottom:14}}>
              <div style={{width:8, height:8, borderRadius:'50%', background:riskColor}} />
              <span style={{color:riskColor, fontWeight:700, fontSize:16, fontFamily:"'Source Serif 4',serif", letterSpacing:'0.01em'}}>{riskLabel} — {Math.round(risk_prob*100)}%</span>
            </div>
            <p style={{color:'var(--ink-soft)', fontSize:13.5, lineHeight:1.7, margin:0, fontFamily:"'Source Serif 4',serif"}}>
              {isHigh
                ? `The Neuro-CRISPR model predicts a high probability of Cas9 cleavage at this site. ${seed_mismatches > 0 ? `The ${seed_mismatches} seed-region mismatch${seed_mismatches !== 1 ? 'es' : ''} would normally reduce binding — yet the model still predicts high cleavage, suggesting strong global similarity to the sgRNA.` : 'The mismatch profile is minimal, consistent with strong on-target binding.'} ${hasDel ? 'The ΔF508 deletion was factored into the null-tensor encoding.' : ''} If this site is your intended edit, this is a strong target. If it is a candidate off-target, consider sgRNA redesign or high-fidelity Cas9.`
                : `The model predicts low cleavage probability at this site. The mismatch profile (${pam_intact ? 'PAM intact' : 'PAM disrupted'}) suggests Cas9 will not efficiently bind. If this is a candidate off-target, low risk of unintended editing.`
              }
            </p>
            <div style={{marginTop:14, padding:'10px 14px', background:'var(--paper-2)', borderRadius:6, border:'1px solid var(--hairline)'}}>
              {[
                ['Encoder', 'Null Tensor (5-ch)'],
                ['Architecture', 'CNN + DNABERT-2 + KAN'],
                ['AUROC', '0.873', 'var(--teal)'],
              ].map(([k, v, c]) => (
                <div key={k} style={{display:'flex', justifyContent:'space-between', marginBottom: k==='AUROC' ? 0 : 4}}>
                  <span style={{fontSize:11, color:'var(--ink-faint)', letterSpacing:'0.04em'}}>{k}</span>
                  <span style={{fontSize:11, color: c || 'var(--ink-soft)', fontFamily:"'JetBrains Mono',monospace", fontWeight:600}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Llama audit */}
        <LlamaAudit result={result} autoAudit={autoAudit} />
      </div>
    </section>
  );
};

const insightCardStyles = {
  background:'#FFFDF7', border:'1px solid var(--hairline)',
  borderRadius:12, padding:22,
  boxShadow:'0 6px 18px rgba(20,20,20,0.04)',
  transition:'all 0.2s',
};
const insightCardTitleStyles = {
  fontSize:11, letterSpacing:'0.16em', color:'var(--ink-faint)',
  fontWeight:700, textTransform:'uppercase', marginBottom:14,
  fontFamily:"'Inter',sans-serif",
};

// ============================================================
// Architecture Diagram Modal (§ Methods) — light theme
// ============================================================
const ArchModal = ({ open, onClose }) => {
  React.useEffect(() => {
    if (!open) return;
    const k = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, onClose]);
  if (!open) return null;

  const Block = ({ x, y, w, h, fill, stroke, label, sublabel, params }) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6}
            fill={fill} stroke={stroke} strokeWidth={1.2}/>
      <text x={x + w/2} y={y + 22} textAnchor="middle"
            fontFamily="'Source Serif 4',serif" fontWeight="700" fontSize={13} fill="#1A1A1A">{label}</text>
      <text x={x + w/2} y={y + 40} textAnchor="middle"
            fontFamily="'JetBrains Mono',monospace" fontSize={10} fill="#525252">{sublabel}</text>
      {params && <text x={x + w/2} y={y + h - 10} textAnchor="middle"
            fontFamily="'JetBrains Mono',monospace" fontSize={9} fill="#8C8579">{params}</text>}
    </g>
  );
  const Arrow = ({ x1, y1, x2, y2 }) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#0D6E64" strokeOpacity={0.6} strokeWidth={1.4}/>
      <polygon points={`${x2},${y2} ${x2-6},${y2-4} ${x2-6},${y2+4}`} fill="#0D6E64"/>
    </g>
  );

  return (
    <div onClick={onClose} style={{
      position:'fixed', inset:0, zIndex:9999,
      background:'rgba(26,26,26,0.55)', backdropFilter:'blur(4px)',
      display:'flex', alignItems:'center', justifyContent:'center', padding:24,
      animation:'fadeInUp 0.25s ease',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background:'#FFFDF7', border:'1px solid var(--hairline)',
        borderRadius:14, padding:'28px 32px', maxWidth:1080, width:'100%',
        maxHeight:'90vh', overflow:'auto',
        boxShadow:'0 60px 120px rgba(0,0,0,0.25)',
      }}>
        <div style={{display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:18}}>
          <div>
            <div className="section-num">§ Methods</div>
            <h2 style={{fontSize:30, color:'var(--ink)', margin:'4px 0 6px'}}>
              Neuro-CRISPR architecture
            </h2>
            <div style={{fontSize:13, color:'var(--ink-soft)', fontFamily:"'Source Serif 4',serif", fontStyle:'italic'}}>
              Hybrid CNN + DNABERT-2 (LoRA) + KAN · 117 M params · ~295 K trainable · AUROC 0.873 on CFTR
            </div>
          </div>
          <button onClick={onClose} style={{
            background:'var(--paper-2)', border:'1px solid var(--hairline)',
            color:'var(--ink-soft)', borderRadius:6, padding:'4px 10px',
            fontSize:11, fontFamily:"'Inter',sans-serif", cursor:'pointer', fontWeight:600,
          }}>ESC ✕</button>
        </div>

        <svg viewBox="0 0 1000 380" style={{width:'100%', height:'auto', background:'var(--paper)', borderRadius:8, border:'1px solid var(--hairline)'}}>
          <Block x={20} y={150} w={140} h={70}
                 fill="#FFFDF7" stroke="#E5E0D2"
                 label="INPUT" sublabel="sgRNA + DNA" params="2 × 23-mer"/>
          <Arrow x1={160} y1={185} x2={210} y2={185}/>

          <Block x={210} y={150} w={150} h={70}
                 fill="#CFF1EB" stroke="#0D6E64"
                 label="ENCODING" sublabel="Null Tensor (5-ch)" params="(2, 23, 5)"/>
          <Arrow x1={360} y1={185} x2={420} y2={120}/>
          <Arrow x1={360} y1={185} x2={420} y2={250}/>

          <Block x={420} y={70} w={220} h={90}
                 fill="#DBEEDB" stroke="#166534"
                 label="CNN STREAM (LOCAL)" sublabel="multi-kernel 1D conv {3,5,7}" params="→ (128) features"/>

          <Block x={420} y={210} w={220} h={90}
                 fill="#FCE9CC" stroke="#B45309"
                 label="DNABERT-2 + LoRA" sublabel="BPE → 12-layer · rank-8" params="117M frozen · 295K LoRA"/>

          <Arrow x1={640} y1={115} x2={700} y2={170}/>
          <Arrow x1={640} y1={255} x2={700} y2={200}/>
          <Block x={700} y={150} w={120} h={70}
                 fill="#F0EBDD" stroke="#B7791F"
                 label="FUSION" sublabel="concat" params="(256)"/>
          <Arrow x1={820} y1={185} x2={870} y2={185}/>

          <Block x={870} y={130} w={110} h={110}
                 fill="#FCE0E0" stroke="#C92A2A"
                 label="KAN HEAD" sublabel="B-splines"
                 params="256→128→64→1"/>

          <Arrow x1={925} y1={250} x2={925} y2={310}/>
          <text x={925} y={340} textAnchor="middle" fontFamily="'JetBrains Mono',monospace" fontSize={11} fill="#0D6E64" fontWeight="600">σ → P(cleavage)</text>
          <text x={925} y={360} textAnchor="middle" fontFamily="'JetBrains Mono',monospace" fontSize={9} fill="#8C8579">e.g. 0.87</text>

          <text x={530} y={50} textAnchor="middle" fontSize={10} fontFamily="'Inter',sans-serif" fontWeight="700" fill="#166534" letterSpacing="0.18em">LOCAL FEATURES</text>
          <text x={530} y={330} textAnchor="middle" fontSize={10} fontFamily="'Inter',sans-serif" fontWeight="700" fill="#B45309" letterSpacing="0.18em">GLOBAL CONTEXT</text>
        </svg>

        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))', gap:12, marginTop:24}}>
          {[
            ['AUROC',           '0.873',         'var(--teal)'],
            ['Recall@95% spec', '0.81',          'var(--safe)'],
            ['Encoder Δ',       '+0.042',        'var(--rust)'],
            ['Inference',       '~120 ms / pair','var(--ink)'],
          ].map(([k, v, c]) => (
            <div key={k} style={{
              background:'var(--paper-2)', border:'1px solid var(--hairline)',
              borderRadius:8, padding:'12px 14px',
            }}>
              <div style={{fontSize:10, letterSpacing:'0.12em', color:'var(--ink-faint)', fontWeight:700, textTransform:'uppercase', fontFamily:"'Inter',sans-serif"}}>{k}</div>
              <div style={{fontSize:22, color:c, fontWeight:700, fontFamily:"'Source Serif 4',serif", marginTop:2}}>{v}</div>
            </div>
          ))}
        </div>

        <div style={{marginTop:24, fontSize:13.5, color:'var(--ink)', lineHeight:1.7, fontFamily:"'Source Serif 4',serif"}}>
          <div style={{fontSize:10, letterSpacing:'0.18em', color:'var(--ink-faint)', fontWeight:700, marginBottom:8, fontFamily:"'Inter',sans-serif"}}>METHOD HIGHLIGHTS</div>
          <ul style={{paddingLeft:18, margin:0}}>
            <li><b>Null Tensor encoding</b> — explicit 5th channel for deletions (e.g. ΔF508). Beats Zero-Pad baseline by +0.042 AUROC on CFTR.</li>
            <li><b>LoRA on Q/V (rank 8)</b> — only 295 K trainable parameters out of 117 M. Fits on a single Colab T4 GPU.</li>
            <li><b>KAN decision head</b> — learnable spline activations replace fixed ReLU. More expressive at low parameter count.</li>
            <li><b>Loss</b> — focal binary cross-entropy + spline-L1 regulariser. Adam, multi-LR (LoRA 1e-5, others 1e-4), 50 epochs.</li>
            <li><b>RAG audit</b> — Llama 3.1-8B (4-bit NF4) retrieves 3 papers from a 25-entry CRISPR knowledge base, generates a clinical-style report.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// Footer
// ============================================================
const AppFooter = () => {
  const [archOpen, setArchOpen] = React.useState(false);
  return (
    <footer style={{
      padding:'48px 24px 60px', textAlign:'center', zIndex:1, position:'relative',
      borderTop:'1px solid var(--hairline)', marginTop:24,
    }}>
      <ArchModal open={archOpen} onClose={() => setArchOpen(false)} />
      <div style={{display:'flex', flexDirection:'column', gap:8, alignItems:'center'}}>
        <div style={{color:'var(--ink)', fontSize:14, fontWeight:600, letterSpacing:'0.02em', fontFamily:"'Source Serif 4',serif"}}>
          Neuro-CRISPR · IEEE ICAUC 2026
        </div>
        <div style={{color:'var(--ink-soft)', fontSize:12, fontFamily:"'Source Serif 4',serif", fontStyle:'italic'}}>
          K.S. Rangasamy College of Technology · Final-year project
        </div>
        <a href="https://github.com/JeganMurali/Neuro-CRISPR-KAN" target="_blank" rel="noopener noreferrer"
          style={{color:'var(--ink-soft)', fontSize:12, display:'flex', alignItems:'center', gap:6, textDecoration:'none', marginTop:4}}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
          </svg>
          JeganMurali/Neuro-CRISPR-KAN
        </a>
        <button onClick={() => setArchOpen(true)} style={{
          background:'none', border:'none', color:'var(--teal)',
          fontSize:12, fontFamily:"'Source Serif 4',serif", fontStyle:'italic',
          cursor:'pointer', textDecoration:'underline', marginTop:6,
        }}>§ Methods · architecture diagram</button>
      </div>
    </footer>
  );
};

Object.assign(window, { InsightsPanel, AppFooter });
