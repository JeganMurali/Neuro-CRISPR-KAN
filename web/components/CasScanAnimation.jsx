
// ============================================================
// Cas9 Scan Animation — cinema (dark) inset on a paper page
// IGV base colors during render · match/mismatch palette on pulse
// ============================================================

const IGV = { A:'#00C853', T:'#DC2626', G:'#B7791F', C:'#2962FF' };
const igvBase = (ch) => IGV[(ch || '').toUpperCase()] || '#71717A';

// Daylight Lab palette (mirrors index.html tokens)
const PAL = {
  teal:    '#14B8A6',   // match (cinema-safe brighter for dark bg)
  mint:    '#34D399',
  danger:  '#F87171',   // mismatch on dark
  gold:    '#FBBF24',   // PAM
  ink:     '#F1F4FA',
  inkSoft: 'rgba(255,255,255,0.55)',
  faint:   'rgba(255,255,255,0.15)',
};

// ── Annotated DNA pair (verdict reveal) ─────────────────────
const AnnotatedDNAPair = ({ sgRNA, dna }) => {
  const N = 23;
  const baseW = 32, gap = 3;
  const totalW = N * (baseW + gap) - gap;
  const seedColor = (i, match) => {
    if (i >= 20) return PAL.gold;
    if (i >= 9 && i <= 19) return match ? PAL.teal : PAL.danger;
    return match ? PAL.mint : PAL.danger;
  };
  const mismatches = Array.from({length:N}).map((_,i) =>
    sgRNA && dna && i < 20 && sgRNA[i]?.toUpperCase() !== dna[i]?.toUpperCase());
  return (
    <div style={{overflowX:'auto', paddingBottom:4}}>
      <svg width={totalW + 20} height={180} style={{display:'block', minWidth: totalW + 20, overflow:'visible'}}>
        <g>
          <path d={`M ${10 + 9*(baseW+gap)} 14 L ${10 + 9*(baseW+gap)} 8 L ${10 + 19*(baseW+gap)+baseW} 8 L ${10 + 19*(baseW+gap)+baseW} 14`}
                fill="none" stroke={PAL.teal} strokeOpacity={0.55} strokeWidth={1}/>
          <text x={10 + (9*(baseW+gap) + 19*(baseW+gap)+baseW)/2} y={5} textAnchor="middle" fontSize={8} fill={PAL.teal}
                fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.18em">SEED</text>
        </g>
        <g>
          <path d={`M ${10 + 20*(baseW+gap)} 14 L ${10 + 20*(baseW+gap)} 8 L ${10 + 22*(baseW+gap)+baseW} 8 L ${10 + 22*(baseW+gap)+baseW} 14`}
                fill="none" stroke={PAL.gold} strokeOpacity={0.7} strokeWidth={1}/>
          <text x={10 + (20*(baseW+gap) + 22*(baseW+gap)+baseW)/2} y={5} textAnchor="middle" fontSize={8} fill={PAL.gold}
                fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.18em">
            {dna && dna.slice(-2).toUpperCase()==='GG' ? 'PAM ✓' : 'PAM ✗'}
          </text>
        </g>
        {Array.from({length:N}).map((_,i) => {
          const ch = sgRNA?.[i]?.toUpperCase() || '·';
          const baseCol = igvBase(ch);
          const x = 10 + i*(baseW+gap);
          return (<g key={`sg-${i}`}>
            <rect x={x} y={18} width={baseW} height={30} rx={5} fill={`${baseCol}20`} stroke={`${baseCol}55`} strokeWidth={1}/>
            <text x={x+baseW/2} y={39} textAnchor="middle" fontSize={15} fontWeight="700" fill={baseCol} fontFamily="'JetBrains Mono',monospace">
              {ch}
            </text>
          </g>);
        })}
        {Array.from({length:N}).map((_,i) => {
          const match = sgRNA?.[i]?.toUpperCase() === dna?.[i]?.toUpperCase();
          const col = seedColor(i, match);
          const x = 10 + i*(baseW+gap) + baseW/2;
          return (<line key={`ln-${i}`} x1={x} y1={50} x2={x} y2={66}
                    stroke={match ? `${col}80` : PAL.danger} strokeWidth={match ? 1 : 1.5}
                    strokeDasharray={match ? 'none' : '2,2'}/>);
        })}
        {Array.from({length:N}).map((_,i) => {
          const ch = dna?.[i]?.toUpperCase() || '·';
          const baseCol = igvBase(ch);
          const x = 10 + i*(baseW+gap);
          return (<g key={`dn-${i}`}>
            <rect x={x} y={68} width={baseW} height={30} rx={5} fill={`${baseCol}18`} stroke={`${baseCol}45`} strokeWidth={1}/>
            <text x={x+baseW/2} y={89} textAnchor="middle" fontSize={15} fontWeight="700" fill={baseCol} fontFamily="'JetBrains Mono',monospace">
              {ch}
            </text>
          </g>);
        })}
        {mismatches.map((isMM, i) => {
          if (!isMM) return null;
          const x = 10 + i*(baseW+gap) + baseW/2;
          return (<g key={`mm-${i}`}>
            <line x1={x} y1={100} x2={x} y2={118} stroke={PAL.danger} strokeWidth={1.5}/>
            <polygon points={`${x},122 ${x-4},114 ${x+4},114`} fill={PAL.danger}/>
            <text x={x} y={138} textAnchor="middle" fontSize={7} fill={PAL.danger} fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.05em">MM</text>
            <text x={x} y={150} textAnchor="middle" fontSize={7} fill={`${PAL.danger}99`} fontFamily="'JetBrains Mono',monospace">{i+1}</text>
          </g>);
        })}
        <text x={6} y={38} textAnchor="end" fontSize={9} fill={PAL.inkSoft} fontFamily="'Inter',sans-serif" fontWeight="600">sg</text>
        <text x={6} y={88} textAnchor="end" fontSize={9} fill={PAL.inkSoft} fontFamily="'Inter',sans-serif" fontWeight="600">dn</text>
      </svg>
    </div>
  );
};

// ── Curved DNA Helix (3D-tilted, sinusoidal strands) ─────────
const CurvedHelix = ({ sgRNA, dna, visibleBases, pulsedBases, scanProgress }) => {
  const N = 23;
  const baseW = 38;
  const padX = 50;
  const totalW = padX * 2 + (N - 1) * baseW;
  const amp = 22;
  const period = 4;
  const cy1 = 60;
  const cy2 = 130;
  const H = 200;

  const strandTop = [];
  const strandBottom = [];
  for (let i = 0; i < N; i++) {
    const x = padX + i * baseW;
    const phase = (i / period) * Math.PI;
    strandTop.push({ x, y: cy1 + Math.sin(phase) * amp });
    strandBottom.push({ x, y: cy2 - Math.sin(phase) * amp });
  }

  const pathFromPoints = (pts) => {
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i-1], curr = pts[i];
      const cx1 = prev.x + (curr.x - prev.x) / 2;
      const cx2 = prev.x + (curr.x - prev.x) / 2;
      d += ` C ${cx1} ${prev.y}, ${cx2} ${curr.y}, ${curr.x} ${curr.y}`;
    }
    return d;
  };

  // IGV base color when rendering; pulse swap to match/mismatch palette
  const baseColor = (i, ch, pulsed) => {
    if (!pulsed) return igvBase(ch);
    if (i >= 20) return PAL.gold;
    const match = sgRNA[i]?.toUpperCase() === dna[i]?.toUpperCase();
    if (i >= 9 && i <= 19) return match ? PAL.teal : PAL.danger;
    return match ? PAL.mint : PAL.danger;
  };

  const scannerX = padX + scanProgress * (N - 1) * baseW;

  return (
    <div style={{
      perspective: 1200,
      width: '100%',
      maxWidth: totalW,
      margin: '0 auto',
      overflowX: 'auto',
    }}>
      <svg
        width={totalW}
        height={H}
        style={{
          transform: 'rotateX(8deg) rotateZ(-1deg)',
          display: 'block',
          minWidth: totalW,
          overflow: 'visible',
        }}
      >
        <defs>
          <linearGradient id="strand-top" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(20,184,166,0.65)"/>
            <stop offset="100%" stopColor="rgba(20,184,166,0.3)"/>
          </linearGradient>
          <linearGradient id="strand-bot" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(251,191,36,0.55)"/>
            <stop offset="100%" stopColor="rgba(251,191,36,0.25)"/>
          </linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="3"/></filter>
        </defs>

        <path d={pathFromPoints(strandTop)} fill="none"
              stroke="url(#strand-top)" strokeWidth={2.5} strokeLinecap="round"
              opacity={visibleBases >= N ? 1 : 0.6}/>

        <path d={pathFromPoints(strandBottom)} fill="none"
              stroke="url(#strand-bot)" strokeWidth={2.5} strokeLinecap="round"
              opacity={visibleBases >= N ? 1 : 0.6}/>

        {/* H-bonds */}
        {strandTop.map((p, i) => {
          if (i >= visibleBases) return null;
          const match = sgRNA[i]?.toUpperCase() === dna[i]?.toUpperCase();
          const pulsed = pulsedBases.includes(i);
          return (
            <line key={`hb-${i}`} x1={p.x} y1={p.y + 14} x2={strandBottom[i].x} y2={strandBottom[i].y - 14}
                  stroke={pulsed ? (match ? `${PAL.mint}b3` : `${PAL.danger}d8`) : 'rgba(255,255,255,0.10)'}
                  strokeWidth={pulsed && !match ? 1.6 : 1}
                  strokeDasharray={match ? 'none' : '3,3'}
                  style={{transition: 'stroke 0.25s'}}/>
          );
        })}

        {/* Top strand bases (sgRNA) */}
        {strandTop.map((p, i) => {
          if (i >= visibleBases) return null;
          const ch = sgRNA[i]?.toUpperCase() || '·';
          const color = baseColor(i, ch, pulsedBases.includes(i));
          const isMM = pulsedBases.includes(i) && sgRNA[i]?.toUpperCase() !== dna[i]?.toUpperCase();
          return (
            <g key={`tb-${i}`}>
              <circle cx={p.x} cy={p.y} r={13} fill={`${color}22`} stroke={`${color}99`} strokeWidth={1.2}
                      style={{filter: isMM ? `drop-shadow(0 0 8px ${PAL.danger})` : 'none', transition:'filter 0.3s'}}/>
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fontWeight="700"
                    fill={color} fontFamily="'JetBrains Mono',monospace">{ch}</text>
            </g>
          );
        })}

        {/* Bottom strand bases (DNA target) */}
        {strandBottom.map((p, i) => {
          if (i >= visibleBases) return null;
          const ch = dna[i]?.toUpperCase() || '·';
          const color = baseColor(i, ch, pulsedBases.includes(i));
          return (
            <g key={`bb-${i}`}>
              <circle cx={p.x} cy={p.y} r={13} fill={`${color}1c`} stroke={`${color}80`} strokeWidth={1.2}/>
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fontWeight="700"
                    fill={color} fontFamily="'JetBrains Mono',monospace">{ch}</text>
            </g>
          );
        })}

        {/* Brackets */}
        {pulsedBases.length >= 19 && (
          <>
            <rect x={padX + 9*baseW - 14} y={4} width={11*baseW} height={H - 8} rx={6}
                  fill="none" stroke={`${PAL.teal}55`} strokeWidth={1} strokeDasharray="3,3"/>
            <text x={padX + 9*baseW - 8} y={H - 4} fontSize={9} fill={PAL.teal} fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.18em">SEED</text>
          </>
        )}
        {pulsedBases.length >= 23 && (
          <>
            <rect x={padX + 20*baseW - 14} y={4} width={3*baseW} height={H - 8} rx={6}
                  fill="none" stroke={`${PAL.gold}80`} strokeWidth={1} strokeDasharray="3,3"/>
            <text x={padX + 20*baseW - 8} y={H - 4} fontSize={9} fill={PAL.gold} fontFamily="'Inter',sans-serif" fontWeight="700" letterSpacing="0.18em">PAM</text>
          </>
        )}

        {/* Cas9 scanner */}
        {scanProgress > 0 && scanProgress < 1.05 && (
          <g transform={`translate(${scannerX}, ${(cy1+cy2)/2})`} style={{transition:'transform 0.06s linear'}}>
            <circle r={26} fill={`${PAL.teal}10`} stroke={`${PAL.teal}99`} strokeWidth={1.5} strokeDasharray="2,3"/>
            <circle r={16} fill={`${PAL.teal}1f`} stroke={PAL.teal} strokeOpacity={0.7} strokeWidth={1}/>
            <circle r={6} fill={PAL.teal} filter="url(#glow)"/>
            <line x1={0} y1={-90} x2={0} y2={90} stroke={`${PAL.teal}88`} strokeWidth={1}
                  style={{filter:`drop-shadow(0 0 8px ${PAL.teal})`}}/>
          </g>
        )}
      </svg>
    </div>
  );
};

// ── Main animation component ────────────────────────────────
const CasScanAnimation = ({ result, onComplete }) => {
  const [stage, setStage] = React.useState(0);
  const [visibleBases, setVisibleBases] = React.useState(0);
  const [scanProgress, setScanProgress] = React.useState(0);
  const [pulsedBases, setPulsedBases] = React.useState([]);
  const [showVerdict, setShowVerdict] = React.useState(false);
  const [paused, setPaused] = React.useState(false);
  const [speed, setSpeed] = React.useState(1);

  const sectionRef = React.useRef(null);
  const timersRef = React.useRef([]);
  const stateRef = React.useRef({ paused: false, speed: 1 });
  React.useEffect(() => { stateRef.current = { paused, speed }; }, [paused, speed]);

  const { sgRNA = '', dna = '', risk_prob = 0, mismatches = 0, seed_mismatches = 0, pam_intact = true } = result || {};
  const N = 23;

  const clearTimers = () => {
    timersRef.current.forEach(id => { clearTimeout(id); clearInterval(id); });
    timersRef.current = [];
  };
  const setT = (fn, ms) => {
    const id = setTimeout(fn, ms / stateRef.current.speed);
    timersRef.current.push(id); return id;
  };

  const runAnimation = React.useCallback(() => {
    clearTimers();
    setStage(0); setVisibleBases(0); setScanProgress(0); setPulsedBases([]); setShowVerdict(false);

    setStage(1);
    const baseStep = (i) => {
      if (i > N) return;
      setVisibleBases(i);
      if (i < N) setT(() => baseStep(i + 1), 1100 / N);
    };
    baseStep(1);

    setT(() => {
      setStage(2);
      const sweepStart = Date.now();
      const tick = () => {
        if (stateRef.current.paused) { setT(tick, 60); return; }
        const elapsed = (Date.now() - sweepStart) * stateRef.current.speed;
        const p = Math.min(elapsed / 1600, 1);
        setScanProgress(p);
        if (p < 1) setT(tick, 30); else setT(stage3, 0);
      };
      tick();
    }, 1100);

    const stage3 = () => {
      setStage(3);
      for (let i = 0; i < N; i++) {
        setT(() => setPulsedBases(prev => prev.includes(i) ? prev : [...prev, i]), i * 80);
      }
      setT(stage4, N * 80 + 200);
    };

    const stage4 = () => {
      setStage(4);
      setT(() => {
        setStage(5);
        setShowVerdict(true);
        if (onComplete) onComplete();
      }, 700);
    };
  }, [result, onComplete]);

  // Play the animation exactly once per new prediction.
  // Scroll-driven replay was removed — re-entering the viewport while
  // scrolling back up should NOT restart the animation.
  React.useEffect(() => {
    if (result) runAnimation();
    return () => clearTimers();
  }, [result]);

  if (!result) return null;

  const riskColor = risk_prob < 0.3 ? PAL.mint : risk_prob < 0.7 ? PAL.gold : PAL.danger;
  const riskLabel = risk_prob < 0.3 ? 'LOW CLEAVAGE PROBABILITY' : risk_prob < 0.7 ? 'MODERATE CLEAVAGE PROBABILITY' : 'HIGH CLEAVAGE PROBABILITY';

  const stageLabel = {
    1: 'Rendering DNA double helix…',
    2: 'Cas9 + sgRNA complex scanning…',
    3: 'Evaluating base-pair binding…',
    4: 'Computing cleavage probability…',
    5: 'Analysis complete.',
  }[stage] || '';

  return (
    <section ref={sectionRef} id="scan-section" style={scanSectionStyles}>
      {/* Section header (paper) */}
      <div style={{maxWidth:1080, width:'100%', margin:'0 auto 16px', padding:'0 8px'}}>
        <div className="section-num">§ II · Live binding analysis</div>
        <h2 style={{fontSize:28, marginTop:4, color:'var(--ink)'}}>Cas9 scanning the target</h2>
        <p style={{color:'var(--ink-soft)', fontSize:14, marginTop:6, fontFamily:"'Source Serif 4',serif"}}>
          A cinematic replay of how the model evaluates this pair, base by base.
        </p>
      </div>

      {/* Cinema inset */}
      <div style={scanInnerStyles}>
        <div style={sectionLabelStyles}>
          <span style={{color: PAL.teal, fontSize:11, letterSpacing:'0.2em', fontWeight:700, fontFamily:"'Inter',sans-serif"}}>STAGE {stage}/5</span>
          <span style={{color: PAL.inkSoft, fontSize:11, marginLeft:12, flex:1, fontFamily:"'Source Serif 4',serif", fontStyle:'italic'}}>{stageLabel}</span>
          <div style={controlsRowStyles}>
            <button style={ctrlBtn(paused)} onClick={() => setPaused(p => !p)} title={paused ? 'Play' : 'Pause'}>
              {paused ? '▶' : '⏸'}
            </button>
            <button style={ctrlBtn(false)} onClick={runAnimation} title="Replay">↺</button>
            {[0.5, 1, 2].map(s => (
              <button key={s} style={ctrlBtn(speed === s)} onClick={() => setSpeed(s)}>{s}×</button>
            ))}
          </div>
        </div>

        <div style={{
          padding: '24px 0 8px',
          opacity: stage >= 1 ? 1 : 0,
          transform: stage >= 4 ? 'scale(0.85) translateY(-10px)' : 'scale(1)',
          transition: 'transform 0.7s cubic-bezier(0.34,1.56,0.64,1), opacity 0.5s',
          filter: stage >= 4 ? 'blur(0.6px)' : 'none',
        }}>
          <CurvedHelix sgRNA={sgRNA} dna={dna} visibleBases={visibleBases} pulsedBases={pulsedBases} scanProgress={scanProgress}/>
        </div>

        {showVerdict && (
          <div style={{
            marginTop: 28,
            background:'rgba(255,255,255,0.04)',
            border:`1px solid ${riskColor}55`,
            borderRadius:14, padding:'26px 30px',
            animation:'slideUpBounce 0.6s cubic-bezier(0.34,1.56,0.64,1)',
            boxShadow:`0 0 60px ${riskColor}1c`,
            position:'relative',
          }}>
            <div style={{
              position:'absolute', top:18, right:18,
              background:`${riskColor}1f`, border:`1px solid ${riskColor}55`,
              borderRadius:6, padding:'5px 12px',
              display:'flex', alignItems:'center', gap:8,
            }}>
              <div style={{width:6, height:6, borderRadius:'50%', background:riskColor, boxShadow:`0 0 6px ${riskColor}`}}/>
              <span style={{color:riskColor, fontWeight:700, fontSize:12, fontFamily:"'Inter',sans-serif", letterSpacing:'0.04em'}}>
                {Math.round(risk_prob*100)}% · {riskLabel}
              </span>
            </div>
            <div style={{fontSize:11, letterSpacing:'0.2em', color:PAL.inkSoft, fontWeight:700, marginBottom:18, fontFamily:"'Inter',sans-serif"}}>
              BINDING ANALYSIS
            </div>
            <AnnotatedDNAPair sgRNA={sgRNA} dna={dna}/>

            <div style={{
              marginTop: 18, padding: '10px 14px', borderRadius: 8,
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.10)',
              fontSize: 12, lineHeight: 1.65, color: PAL.inkSoft,
              fontFamily: "'Source Serif 4', serif",
            }}>
              <span style={{color:'rgba(255,255,255,0.65)', letterSpacing:'0.16em', fontWeight:700, fontSize:10, marginRight:8, fontFamily:"'Inter',sans-serif"}}>HOW TO READ THIS</span>
              The score is the probability Cas9 cleaves at this site.
              {' '}
              <span style={{color: PAL.mint}}>If this is your <b>intended target</b> → high = effective edit ✓</span>
              {' · '}
              <span style={{color: PAL.danger}}>If this is a <b>candidate off-target</b> → high = unintended cut ⚠</span>
            </div>

            <div style={{display:'flex', gap:10, marginTop:22, flexWrap:'wrap'}}>
              {[
                [`${mismatches} mismatch${mismatches!==1?'es':''}`, mismatches > 2 ? PAL.danger : PAL.gold],
                [`${seed_mismatches} in seed region`, seed_mismatches > 0 ? PAL.danger : PAL.mint],
                [`PAM ${pam_intact ? '✓ intact' : '✗ disrupted'}`, pam_intact ? PAL.mint : PAL.danger],
              ].map(([label, color]) => (
                <span key={label} style={{background:`${color}1a`, border:`1px solid ${color}55`, borderRadius:4, padding:'4px 12px', fontSize:12, color, fontWeight:600, fontFamily:"'Inter',sans-serif"}}>
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

// ── Styles ─────────────────────────────────────────────
const scanSectionStyles = {
  display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
  padding:'48px 24px', zIndex:1, position:'relative',
};
const scanInnerStyles = {
  width:'100%', maxWidth:1080,
  background: `linear-gradient(180deg, var(--inset-bg) 0%, var(--inset-bg-2) 100%)`,
  border:'1px solid rgba(255,255,255,0.08)',
  borderRadius:16, padding:'30px 36px',
  boxShadow:'0 30px 80px rgba(14,20,36,0.35), inset 0 1px 0 rgba(255,255,255,0.04)',
  color: PAL.ink,
};
const sectionLabelStyles = {
  display:'flex', alignItems:'center', marginBottom:10, gap:12,
  paddingBottom:14, borderBottom:'1px solid rgba(255,255,255,0.06)',
  flexWrap:'wrap',
};
const controlsRowStyles = { display:'flex', gap:6, alignItems:'center' };
const ctrlBtn = (active) => ({
  background: active ? `${PAL.teal}26` : 'rgba(255,255,255,0.05)',
  border: `1px solid ${active ? `${PAL.teal}66` : 'rgba(255,255,255,0.10)'}`,
  borderRadius: 4,
  padding: '4px 10px',
  color: active ? PAL.teal : PAL.inkSoft,
  fontFamily: "'Inter', sans-serif",
  fontSize: 11, fontWeight: 600,
  cursor: 'pointer',
  transition: 'all 0.15s',
  minWidth: 32,
});

Object.assign(window, { CasScanAnimation });
