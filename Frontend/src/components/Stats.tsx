import { motion, useInView } from "framer-motion";
import { useRef, useEffect, useState } from "react";
import { TrendingUp, Users, DollarSign, Target, Award } from "lucide-react";

const metrics = [
  {
    value: 4.2,
    prefix: "$",
    suffix: "M",
    label: "Paid to creators",
    subtext: "Last 90 days",
    icon: DollarSign,
    color: "violet",
  },
  {
    value: 89,
    suffix: "%",
    label: "Campaign completion",
    subtext: "Industry avg: 64%",
    icon: Target,
    color: "mint",
  },
  {
    value: 3.2,
    suffix: "x",
    label: "Avg. ROAS",
    subtext: "Return on ad spend",
    icon: TrendingUp,
    color: "rose",
  },
  {
    value: 2.4,
    suffix: "k",
    label: "Active creators",
    subtext: "Verified profiles",
    icon: Users,
    color: "gold",
  },
];

const colorMap = {
  violet: {
    bg: "bg-violet-soft",
    text: "text-violet",
    gradient: "from-violet to-violet-glow",
  },
  mint: {
    bg: "bg-mint-soft",
    text: "text-mint",
    gradient: "from-mint to-primary",
  },
  rose: {
    bg: "bg-rose-soft",
    text: "text-rose",
    gradient: "from-rose to-rose-glow",
  },
  gold: {
    bg: "bg-gold-soft",
    text: "text-gold",
    gradient: "from-gold to-amber-400",
  },
};

function AnimatedNumber({ value, prefix = "", suffix = "" }: { value: number; prefix?: string; suffix?: string }) {
  const [displayValue, setDisplayValue] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  useEffect(() => {
    if (isInView) {
      const duration = 2000;
      const steps = 60;
      const increment = value / steps;
      let current = 0;

      const timer = setInterval(() => {
        current += increment;
        if (current >= value) {
          setDisplayValue(value);
          clearInterval(timer);
        } else {
          setDisplayValue(current);
        }
      }, duration / steps);

      return () => clearInterval(timer);
    }
  }, [isInView, value]);

  return (
    <span ref={ref} className="metric-value">
      {prefix}{displayValue.toFixed(value % 1 === 0 ? 0 : 1)}{suffix}
    </span>
  );
}

const Stats = () => {
  return (
    <section className="py-24 sm:py-32 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 bg-gradient-mesh opacity-50" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-gradient-radial opacity-30" />

      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary mb-6">
            <Award className="w-3.5 h-3.5 text-violet" />
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">The Numbers</span>
          </div>
          <h2 className="font-display text-display-sm sm:text-display-md font-normal mb-4 text-balance">
            Real results, real <span className="italic text-violet">impact</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-lg mx-auto">
            Numbers that speak for themselves—transparency you can trust.
          </p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 max-w-6xl mx-auto">
          {metrics.map((metric, index) => {
            const colors = colorMap[metric.color as keyof typeof colorMap];

            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
                className="group"
              >
                <div className="relative h-full p-6 rounded-3xl bg-card border border-border/60 hover:border-border hover:shadow-xl transition-all duration-500 overflow-hidden">
                  {/* Subtle gradient overlay on hover */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${colors.gradient} opacity-0 group-hover:opacity-[0.03] transition-opacity duration-500`} />

                  {/* Icon */}
                  <div className={`w-12 h-12 rounded-2xl ${colors.bg} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300`}>
                    <metric.icon className={`w-5 h-5 ${colors.text}`} />
                  </div>

                  {/* Value */}
                  <p className="font-display text-4xl sm:text-5xl font-normal text-foreground mb-2 tracking-tight">
                    <AnimatedNumber value={metric.value} prefix={metric.prefix} suffix={metric.suffix} />
                  </p>

                  {/* Label */}
                  <p className="font-semibold text-foreground mb-1">{metric.label}</p>

                  {/* Subtext with indicator */}
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full bg-gradient-to-r ${colors.gradient}`} />
                    <p className="text-sm text-muted-foreground">{metric.subtext}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Bottom trust indicator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="flex justify-center mt-12"
        >
          <div className="inline-flex items-center gap-3 px-5 py-3 rounded-full glass-card text-sm">
            <div className="flex -space-x-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="w-6 h-6 rounded-full border-2 border-white bg-gradient-to-br from-violet-400 to-rose-400" />
              ))}
            </div>
            <span className="text-muted-foreground">
              Trusted by <span className="font-semibold text-foreground">500+</span> brands worldwide
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Stats;