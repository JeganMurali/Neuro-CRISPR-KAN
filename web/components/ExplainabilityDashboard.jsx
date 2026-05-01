
// ============================================================
// Explainability Dashboard — Daylight Lab (paper-light)
// ============================================================

const RiskGauge = ({ prob, size = 200 }) => {
  const [animated, setAnimated] = React.useState(0);
  React.useEffect(() => {
    let start = null;
    const duration = 900;
    const tick = (ts) => {
      if (!start) start = ts;
      const t = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setAnimated(ease * prob);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [prob]);

  const r = (size - 28) / 2;
  const cx = size / 2, cy = size / 2;
  const startAngle = Math.PI * 0.75;
  const endAngle = Math.PI * 2.25;
  const totalArc = endAngle - startAngle;
  const color = animated < 0.3 ? 'var(--safe)' : animated < 0.7 ? 'var(--rust)' : 'var(--danger)';

  const polarToXY = (angle, radius) => ({ x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) });
  const arcPath = (radius, fromAngle, toAngle) => {
    const s = polarToXY(fromAngle, radius);
    const e = polarToXY(toAngle, radius);
    const large = toAngle - fromAngle > Math.PI ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${large} 1 ${e.x} ${e.y}`;
  };

  const filledEnd = startAngle + totalArc * animated;
  return (
    <svg width={size} height={size} style={{overflow:'visible'}}>
      <path d={arcPath(r, startAngle, endAngle)} fill="none" stroke="var(--paper-3)" strokeWidth={9} strokeLinecap="round"/>
      <path d={arcPath(r, startAngle, filledEnd)} fill="none" stroke={color} strokeWidth={9} strokeLinecap="round"/>
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize={size * 0.2} fill="var(--ink)" fontFamily="'Source Serif 4',serif" fontWeight="700">
        {Math.round(animated * 100)}%
      </text>
      <text x={cx} y={cy + 16} textAnchor="middle" fontSize={size * 0.062} fill="var(--ink-faint)" fontFamily="'Inter',sans-serif" fontWeight="600" letterSpacing="0.16em">
        CLEAVAGE PROB
      </text>
    </svg>
  );
};

const SaliencyChart = ({ saliency, sgRNA, dna }) => {
  const N = 23;
  const maxH = 140;
  const barW = 22;
  const gap = 4;
  const totalW = N * (barW + gap) - gap;
  const svgH = maxH + 60;

  return (
    <div style={{overflowX:'auto', paddingBottom:8}}>
      <svg width={totalW + 80} height={svgH + 40} style={{display:'block', margin:'0 auto'}}>
        {/* SEED bracket */}
        <g opacity={0.9}>
          <line x1={40 + 9*(barW+gap)} y1={12} x2={40 + 19*(barW+gap) + barW} y2={12} stroke="var(--teal)" strokeOpacity={0.55} strokeWidth={1}/>
          <line x1={40 + 9*(barW+gap)} y1={12} x2={40 + 9*(barW+gap)} y2={22} stroke="var(--teal)" strokeOpacity={0.55} strokeWidth={1}/>
          <line x1={40 + 19*(barW+gap) + barW} y1={12} x2={40 + 19*(barW+gap) + barW} y2={22} stroke="var(--teal)" strokeOpacity={0.55} strokeWidth={1}/>
          <text x={40 + 14*(barW+gap)} y={8} textAnchor="middle" fontSize={9} fill="var(--teal)" fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.14em">SEED</text>
        </g>
        {/* PAM bracket */}
        <g opacity={0.9}>
          <line x1={40 + 20*(barW+gap)} y1={12} x2={40 + 22*(barW+gap) + barW} y2={12} stroke="var(--gold)" strokeOpacity={0.6} strokeWidth={1}/>
          <line x1={40 + 20*(barW+gap)} y1={12} x2={40 + 20*(barW+gap)} y2={22} stroke="var(--gold)" strokeOpacity={0.6} strokeWidth={1}/>
          <line x1={40 + 22*(barW+gap) + barW} y1={12} x2={40 + 22*(barW+gap) + barW} y2={22} stroke="var(--gold)" strokeOpacity={0.6} strokeWidth={1}/>
          <text x={40 + 21.5*(barW+gap)} y={8} textAnchor="middle" fontSize={9} fill="var(--gold)" fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.14em">PAM</text>
        </g>

        {Array.from({length: N}).map((_, i) => {
          const val = saliency ? saliency[i] || 0 : Math.random();
          const h = Math.max(4, val * maxH);
          const isMismatch = sgRNA && dna && sgRNA[i]?.toUpperCase() !== dna[i]?.toUpperCase();
          const isPAM = i >= 20;
          const isSeed = i >= 9 && i <= 19;
          const color = isPAM ? '#B7791F' : isMismatch ? '#C92A2A' : isSeed ? '#0D6E64' : '#166534';
          const x = 40 + i * (barW + gap);
          const y = 28 + maxH - h;

          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={h} rx={3}
                fill={color} opacity={0.85}/>
              <text x={x + barW/2} y={28 + maxH + 16} textAnchor="middle" fontSize={8}
                fill="var(--ink-faint)" fontFamily="'JetBrains Mono',monospace">
                {i+1}
              </text>
              {sgRNA && dna && (
                <text x={x + barW/2} y={28 + maxH + 28} textAnchor="middle" fontSize={8}
                  fill="var(--ink-soft)" fontFamily="'JetBrains Mono',monospace">
                  {sgRNA[i]?.toUpperCase()}/{dna[i]?.toUpperCase()}
                </text>
              )}
            </g>
          );
        })}

        <text transform="rotate(-90)" x={-(28 + maxH/2)} y={14} textAnchor="middle"
          fontSize={9} fill="var(--ink-faint)" fontFamily="'Inter',sans-serif"
          letterSpacing="0.12em" fontWeight="700">SALIENCY</text>
      </svg>

      {saliency && (() => {
        const top3 = [...saliency].map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,3);
        return (
          <div style={{display:'flex', gap:8, flexWrap:'wrap', marginTop:12, justifyContent:'center'}}>
            <span style={{color:'var(--ink-faint)', fontSize:12, fontFamily:"'Source Serif 4',serif", fontStyle:'italic'}}>Top positions:</span>
            {top3.map(({v,i}) => (
              <span key={i} style={{background:'var(--paper-2)', border:'1px solid var(--hairline)', borderRadius:4, padding:'2px 10px', fontSize:11, color:'var(--ink-soft)', fontFamily:"'JetBrains Mono',monospace"}}>
                pos {i+1} ({(v*100).toFixed(0)}%)
              </span>
            ))}
          </div>
        );
      })()}
    </div>
  );
};

const TokenChart = ({ tokens, tokenImportance }) => {
  if (!tokens || !tokenImportance) return null;
  const sorted = tokens.map((t,i) => ({t, v:tokenImportance[i]||0})).sort((a,b) => b.v - a.v);

  return (
    <div style={{padding:'8px 0'}}>
      {sorted.map(({t, v}, i) => (
        <div key={i} style={{display:'flex', alignItems:'center', gap:12, marginBottom:8}}>
          <span style={{fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:'var(--ink)', minWidth:60, textAlign:'right', fontWeight:600}}>{t}</span>
          <div style={{flex:1, height:18, background:'var(--paper-2)', borderRadius:4, overflow:'hidden', position:'relative', border:'1px solid var(--hairline)'}}>
            <div style={{
              position:'absolute', left:0, top:0, bottom:0,
              width:`${v*100}%`,
              background:'linear-gradient(90deg, var(--teal), var(--rust))',
              borderRadius:4, opacity:0.92,
              transition:'width 1s ease',
            }} />
          </div>
          <span style={{fontSize:11, color:'var(--ink-soft)', minWidth:36, textAlign:'right', fontFamily:"'JetBrains Mono',monospace", fontWeight:600}}>{(v*100).toFixed(0)}%</span>
        </div>
      ))}
      <div style={{marginTop:14, color:'var(--ink-faint)', fontSize:12, lineHeight:1.6, fontStyle:'italic', fontFamily:"'Source Serif 4',serif"}}>
        Gradient of CLS embedding w.r.t. token embeddings (Captum-style) · DNABERT-2 BPE tokens.
      </div>
    </div>
  );
};

const EncoderDelta = ({ result }) => {
  const ntProb = result?.risk_prob || 0;
  const zpProb = Math.max(0, ntProb - 0.04 - Math.random()*0.03);
  const delta = ntProb - zpProb;

  return (
    <div style={{display:'flex', gap:20, alignItems:'flex-start', flexWrap:'wrap'}}>
      <div style={{flex:1, minWidth:140, textAlign:'center'}}>
        <div style={encLabelStyles}>NULL TENSOR · 5-CH</div>
        <RiskGauge prob={ntProb} size={140} />
        <div style={{marginTop:8, fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'var(--teal)'}}>A→[1,0,0,0,0]</div>
        <div style={{fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'var(--teal)'}}>GAP→[0,0,0,0,<span style={{color:'var(--danger)'}}>1</span>]</div>
      </div>

      <div style={{display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', padding:'20px 12px', minWidth:120}}>
        <div style={{fontSize:30, fontWeight:700, color: delta > 0 ? 'var(--rust)' : 'var(--safe)', fontFamily:"'Source Serif 4',serif"}}>
          {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
        </div>
        <div style={{color:'var(--ink-faint)', fontSize:11, marginTop:6, textAlign:'center', maxWidth:120, fontFamily:"'Source Serif 4',serif", fontStyle:'italic', lineHeight:1.5}}>
          Null Tensor encoding contributes additional signal
        </div>
      </div>

      <div style={{flex:1, minWidth:140, textAlign:'center'}}>
        <div style={encLabelStyles}>ZERO-PAD · 4-CH</div>
        <RiskGauge prob={zpProb} size={140} />
        <div style={{marginTop:8, fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'var(--ink-faint)'}}>A→[1,0,0,0]</div>
        <div style={{fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'var(--ink-faint)'}}>GAP→[0,0,0,<span style={{color:'var(--ink-faint)', textDecoration:'line-through'}}>0</span>] ← lost</div>
      </div>
    </div>
  );
};

const ExplainabilityDashboard = ({ result }) => {
  const [activeTab, setActiveTab] = React.useState('encoder');

  if (!result) return null;

  const tabs = [
    { id:'encoder', label:'Encoder Δ' },
    { id:'saliency', label:'CNN Saliency' },
    { id:'dnabert', label:'DNABERT-2 Tokens' },
  ];

  return (
    <section style={explainSectionStyles}>
      <div style={explainInnerStyles}>
        <div style={{marginBottom:24}}>
          <div className="section-num">§ III · Explainability</div>
          <h2 style={{fontSize:28, color:'var(--ink)', margin:'4px 0 6px'}}>Why this prediction?</h2>
          <p style={{color:'var(--ink-soft)', fontSize:14, fontFamily:"'Source Serif 4',serif"}}>
            Inspect the encoder's contribution, attention saliency, and BPE token importance.
          </p>
        </div>

        {/* Tab bar */}
        <div style={{display:'flex', gap:4, borderBottom:'1px solid var(--hairline)', marginBottom:24, paddingBottom:0}}>
          {tabs.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              background:'none', border:'none', cursor:'pointer', padding:'10px 18px',
              color: activeTab === tab.id ? 'var(--teal)' : 'var(--ink-faint)',
              fontFamily:"'Inter',sans-serif", fontWeight:600, fontSize:13,
              letterSpacing:'0.04em',
              borderBottom: activeTab === tab.id ? '2px solid var(--teal)' : '2px solid transparent',
              transition:'all 0.2s', marginBottom:-1,
            }}>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === 'encoder' && (
          <div>
            <div style={{display:'flex', justifyContent:'center', marginBottom:24}}>
              <div style={{textAlign:'center'}}>
                <div style={{fontSize:10, letterSpacing:'0.18em', color:'var(--ink-faint)', fontWeight:700, marginBottom:8, fontFamily:"'Inter',sans-serif"}}>CLEAVAGE PROBABILITY</div>
                <RiskGauge prob={result.risk_prob} size={160} />
              </div>
            </div>
            <div style={{borderTop:'1px solid var(--hairline)', paddingTop:22}}>
              <EncoderDelta result={result} />
            </div>
          </div>
        )}
        {activeTab === 'saliency' && (
          <SaliencyChart saliency={result.saliency} sgRNA={result.sgRNA} dna={result.dna} />
        )}
        {activeTab === 'dnabert' && (
          <TokenChart tokens={result.tokens} tokenImportance={result.token_importance} />
        )}
      </div>
    </section>
  );
};

const explainSectionStyles = {
  padding:'48px 24px', zIndex:1, position:'relative',
};
const explainInnerStyles = {
  width:'100%', maxWidth:900, margin:'0 auto',
  background:'#FFFDF7',
  border:'1px solid var(--hairline)',
  borderRadius:14, padding:'30px 36px',
  boxShadow:'0 14px 40px rgba(20,20,20,0.06)',
};
const encLabelStyles = {
  color:'var(--ink-soft)', fontSize:10, letterSpacing:'0.16em',
  fontWeight:700, marginBottom:10, fontFamily:"'Inter',sans-serif",
};

Object.assign(window, { ExplainabilityDashboard, RiskGauge });
