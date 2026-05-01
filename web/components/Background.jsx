
// Daylight Lab — drifting A T G C letters in IGV colors at low opacity over paper
const Background = () => {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    const particles = [];
    const BASES = [
      { ch: 'A', color: '0,200,83'   },
      { ch: 'T', color: '220,38,38'  },
      { ch: 'G', color: '183,121,31' },
      { ch: 'C', color: '41,98,255'  },
    ];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
      const b = BASES[Math.floor(Math.random() * 4)];
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        base: b.ch,
        color: b.color,
        size: Math.random() * 12 + 9,
        opacity: Math.random() * 0.05 + 0.025,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.12,
        rotation: Math.random() * Math.PI * 2,
        vr: (Math.random() - 0.5) * 0.002,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.vr;
        if (p.x < -30) p.x = canvas.width + 30;
        if (p.x > canvas.width + 30) p.x = -30;
        if (p.y < -30) p.y = canvas.height + 30;
        if (p.y > canvas.height + 30) p.y = -30;

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.font = `600 ${p.size}px 'JetBrains Mono', monospace`;
        ctx.fillStyle = `rgba(${p.color}, ${p.opacity})`;
        ctx.fillText(p.base, 0, 0);
        ctx.restore();
      });
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return React.createElement('canvas', {
    ref: canvasRef,
    style: {
      position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
      pointerEvents: 'none', zIndex: 0,
    }
  });
};

Object.assign(window, { Background });
