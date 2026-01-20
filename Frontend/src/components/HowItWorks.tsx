import { Share2, Crosshair, BarChart, ShieldCheck, FileCheck } from "lucide-react";

// NOTE: This is the homepage component, not the dedicated page.
const HowItWorks = () => {
  return (
    <section className="py-32 bg-[#0A0A0A] text-white relative overflow-hidden">
      {/* Tech Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#0A0A0A] via-transparent to-[#0A0A0A] pointer-events-none" />

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-20 max-w-3xl mx-auto">
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-6">How Sentinel Secures AI-Written Code</h2>
          <p className="text-gray-400 text-lg">System thinking, not just pattern matching. Sentinel builds a living graph of your security posture.</p>
        </div>

        <div className="relative max-w-5xl mx-auto">
          {/* Central Circuit Line */}
          <div className="absolute left-[28px] md:left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary/20 via-primary/50 to-primary/20 md:-translate-x-1/2" />

          {[
            {
              step: "01",
              title: "Semantic Mapping",
              desc: "Sentinel builds a living security graph of your project: data flows, trust boundaries, and dependencies.",
              icon: <Share2 className="w-5 h-5" />
            },
            {
              step: "02",
              title: "Threat Simulation",
              desc: "It reasons like an attacker. Identifies entry points, simulates abuse paths, and explores privilege escalation.",
              icon: <Crosshair className="w-5 h-5" />
            },
            {
              step: "03",
              title: "Vulnerability Ranking",
              desc: "Findings are prioritized by exploitability, blast radius, and the likelihood of AI reintroduction.",
              icon: <BarChart className="w-5 h-5" />
            },
            {
              step: "04",
              title: "Secure Fixes",
              desc: "Sentinel prepares fixes that are minimal, production-safe, and style-preserving. No refactors. No surprises.",
              icon: <ShieldCheck className="w-5 h-5" />
            },
            {
              step: "05",
              title: "Test Enforcement",
              desc: "For every fix, the exploit is reproduced, the patch applied, and a regression test added.",
              icon: <FileCheck className="w-5 h-5" />
            }
          ].map((item, index) => (
            <div key={item.step} className={`relative flex items-center gap-8 mb-20 last:mb-0 ${index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'}`}>

              {/* Circuit Marker */}
              <div className="absolute left-0 md:left-1/2 w-14 h-14 rounded-full bg-[#0A0A0A] border-4 border-gray-800 flex items-center justify-center md:-translate-x-1/2 z-10 shrink-0 shadow-[0_0_20px_rgba(0,166,118,0.2)]">
                <div className="text-primary">{item.icon}</div>
              </div>

              {/* Spacer for Desktop Layout alignment */}
              <div className="hidden md:block w-1/2" />

              {/* Floating Glass Panel */}
              <div className="flex-1 pl-20 md:pl-0">
                <div className={`relative bg-white/5 border border-white/10 p-8 rounded-2xl backdrop-blur-md hover:bg-white/10 transition-colors group ${index % 2 === 0 ? 'md:mr-16' : 'md:ml-16'}`}>
                  {/* Circuit Connection Line (Horizontal) */}
                  <div className={`hidden md:block absolute top-1/2 w-16 h-0.5 bg-gray-800 ${index % 2 === 0 ? '-right-16' : '-left-16'}`} />

                  <span className="text-xs font-bold text-primary uppercase tracking-widest mb-3 block">Step {item.step}</span>
                  <h3 className="font-bold text-2xl mb-4 text-white group-hover:text-primary transition-colors">{item.title}</h3>
                  <p className="text-gray-400 leading-relaxed text-base">{item.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;