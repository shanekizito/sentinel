import { motion } from "framer-motion";

const OrbitalIllustration = () => {
  return (
    <div className="relative w-full h-[500px] flex items-center justify-center">
      {/* Background grid */}
      <svg 
        className="absolute inset-0 w-full h-full opacity-30"
        viewBox="0 0 400 400"
        preserveAspectRatio="xMidYMid slice"
      >
        {/* Orbital lines */}
        <ellipse cx="200" cy="200" rx="180" ry="60" fill="none" stroke="hsl(150 10% 60%)" strokeWidth="0.5" strokeDasharray="4 4" />
        <ellipse cx="200" cy="200" rx="140" ry="45" fill="none" stroke="hsl(150 10% 60%)" strokeWidth="0.5" strokeDasharray="4 4" />
        <ellipse cx="200" cy="200" rx="100" ry="32" fill="none" stroke="hsl(150 10% 60%)" strokeWidth="0.5" strokeDasharray="4 4" />
        
        {/* Vertical grid lines */}
        {[100, 150, 200, 250, 300].map((x) => (
          <line key={`v-${x}`} x1={x} y1="100" x2={x} y2="300" stroke="hsl(150 10% 70%)" strokeWidth="0.3" />
        ))}
        
        {/* Horizontal lines */}
        {[150, 200, 250].map((y) => (
          <line key={`h-${y}`} x1="100" y1={y} x2="300" y2={y} stroke="hsl(150 10% 70%)" strokeWidth="0.3" />
        ))}
      </svg>

      {/* Main orbital diagram */}
      <svg 
        className="relative w-full h-full max-w-[450px]"
        viewBox="0 0 400 400"
        fill="none"
      >
        {/* Central core - security nucleus */}
        <motion.g
          animate={{ rotate: 360 }}
          transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Core rings */}
          <circle cx="200" cy="200" r="35" stroke="hsl(152 45% 28%)" strokeWidth="1.5" fill="none" />
          <circle cx="200" cy="200" r="25" stroke="hsl(152 45% 28%)" strokeWidth="1" fill="hsl(152 45% 28% / 0.08)" />
          <circle cx="200" cy="200" r="15" fill="hsl(152 45% 28%)" />
          
          {/* Core cross */}
          <line x1="200" y1="175" x2="200" y2="225" stroke="hsl(152 45% 28%)" strokeWidth="1" />
          <line x1="175" y1="200" x2="225" y2="200" stroke="hsl(152 45% 28%)" strokeWidth="1" />
        </motion.g>

        {/* Outer orbital track 1 */}
        <ellipse 
          cx="200" 
          cy="200" 
          rx="150" 
          ry="55" 
          stroke="hsl(150 20% 25%)" 
          strokeWidth="1" 
          fill="none"
          strokeDasharray="2 6"
        />

        {/* Outer orbital track 2 - rotated */}
        <ellipse 
          cx="200" 
          cy="200" 
          rx="130" 
          ry="48" 
          stroke="hsl(150 10% 60%)" 
          strokeWidth="0.75" 
          fill="none"
          transform="rotate(45 200 200)"
        />

        {/* Outer orbital track 3 */}
        <ellipse 
          cx="200" 
          cy="200" 
          rx="110" 
          ry="40" 
          stroke="hsl(150 10% 60%)" 
          strokeWidth="0.75" 
          fill="none"
          transform="rotate(-30 200 200)"
        />

        {/* Orbiting code symbols */}
        <motion.g
          animate={{ rotate: 360 }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Code block 1 */}
          <g transform="translate(340, 185)">
            <rect x="0" y="0" width="32" height="32" rx="6" stroke="hsl(150 20% 25%)" strokeWidth="1" fill="hsl(45 30% 96%)" />
            <text x="16" y="21" textAnchor="middle" fontSize="14" fontFamily="JetBrains Mono" fill="hsl(150 20% 25%)">{`{}`}</text>
          </g>
        </motion.g>

        <motion.g
          animate={{ rotate: -360 }}
          transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Code block 2 */}
          <g transform="translate(55, 170)">
            <rect x="0" y="0" width="28" height="28" rx="5" stroke="hsl(150 20% 25%)" strokeWidth="1" fill="hsl(45 30% 96%)" />
            <text x="14" y="19" textAnchor="middle" fontSize="12" fontFamily="JetBrains Mono" fill="hsl(150 20% 25%)">&lt;/&gt;</text>
          </g>
        </motion.g>

        <motion.g
          animate={{ rotate: 360 }}
          transition={{ duration: 35, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Code block 3 - Function */}
          <g transform="translate(290, 95)">
            <circle cx="16" cy="16" r="16" stroke="hsl(150 20% 25%)" strokeWidth="1" fill="hsl(45 30% 96%)" />
            <text x="16" y="20" textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono" fill="hsl(150 20% 25%)">fn</text>
          </g>
        </motion.g>

        <motion.g
          animate={{ rotate: -360 }}
          transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Code block 4 - API */}
          <g transform="translate(85, 280)">
            <rect x="0" y="0" width="30" height="24" rx="4" stroke="hsl(150 20% 25%)" strokeWidth="1" fill="hsl(45 30% 96%)" />
            <text x="15" y="16" textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono" fill="hsl(150 20% 25%)">API</text>
          </g>
        </motion.g>

        <motion.g
          animate={{ rotate: 360 }}
          transition={{ duration: 28, repeat: Infinity, ease: "linear" }}
          style={{ transformOrigin: "200px 200px" }}
        >
          {/* Shield - Security node */}
          <g transform="translate(280, 280)">
            <path 
              d="M16 4 L28 10 L28 18 C28 24 22 29 16 32 C10 29 4 24 4 18 L4 10 Z" 
              stroke="hsl(152 45% 28%)" 
              strokeWidth="1.5" 
              fill="hsl(152 45% 28% / 0.15)"
            />
            <circle cx="16" cy="16" r="4" fill="hsl(152 45% 28%)" />
          </g>
        </motion.g>

        {/* Detection nodes - green dots */}
        <motion.circle 
          cx="200" cy="145" r="4" 
          fill="hsl(152 45% 28%)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle 
          cx="260" cy="175" r="3" 
          fill="hsl(152 45% 35%)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
        />
        <motion.circle 
          cx="140" cy="225" r="3" 
          fill="hsl(152 45% 35%)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
        <motion.circle 
          cx="200" cy="255" r="4" 
          fill="hsl(152 45% 28%)"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut", delay: 0.3 }}
        />

        {/* Connection lines from core to nodes */}
        <line x1="200" y1="180" x2="200" y2="149" stroke="hsl(152 45% 28% / 0.3)" strokeWidth="1" strokeDasharray="2 2" />
        <line x1="215" y1="190" x2="257" y2="175" stroke="hsl(152 45% 28% / 0.3)" strokeWidth="1" strokeDasharray="2 2" />
        <line x1="185" y1="210" x2="143" y2="225" stroke="hsl(152 45% 28% / 0.3)" strokeWidth="1" strokeDasharray="2 2" />
        <line x1="200" y1="220" x2="200" y2="251" stroke="hsl(152 45% 28% / 0.3)" strokeWidth="1" strokeDasharray="2 2" />
      </svg>

      {/* Corner decorations */}
      <div className="absolute top-8 left-8">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <line x1="20" y1="0" x2="20" y2="40" stroke="hsl(150 10% 60%)" strokeWidth="1" />
          <line x1="0" y1="20" x2="40" y2="20" stroke="hsl(150 10% 60%)" strokeWidth="1" />
        </svg>
      </div>
      <div className="absolute top-8 right-8">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <line x1="20" y1="0" x2="20" y2="40" stroke="hsl(150 10% 60%)" strokeWidth="1" />
          <line x1="0" y1="20" x2="40" y2="20" stroke="hsl(150 10% 60%)" strokeWidth="1" />
        </svg>
      </div>
    </div>
  );
};

export default OrbitalIllustration;