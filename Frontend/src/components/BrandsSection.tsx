import { motion } from "framer-motion";
import { CheckCircle2, Target, LineChart, Shield, Brain, BarChart3, Building2, TrendingUp } from "lucide-react";

const benefits = [
  { icon: LineChart, text: "Real-time conversion tracking" },
  { icon: BarChart3, text: "Verified creator performance scores" },
  { icon: Shield, text: "Escrow protection for your budget" },
  { icon: Brain, text: "AI-powered creator matching" },
  { icon: Target, text: "Detailed ROI analytics" },
];

const BrandsSection = () => {
  return (
    <section id="brands" className="py-24 sm:py-32 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-mesh opacity-30" />
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 lg:gap-20 items-center">
          {/* Visual - Campaign Dashboard */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="relative order-2 lg:order-1"
          >
            {/* Glow effect */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-rose/10 rounded-full blur-[80px]" />
            
            <div className="relative glass-card-strong rounded-3xl p-8 hover-lift">
              {/* Campaign header */}
              <div className="flex items-start justify-between mb-8">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center shadow-md">
                      <Building2 className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Campaign</p>
                      <p className="font-display text-xl font-normal">Summer Collection Launch</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-mint-soft text-mint text-xs font-semibold">
                  <div className="w-1.5 h-1.5 rounded-full bg-mint animate-pulse" />
                  Active
                </div>
              </div>

              {/* Progress */}
              <div className="mb-8">
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-muted-foreground">Budget spent</span>
                  <span className="font-semibold">$8,400 / $12,000</span>
                </div>
                <div className="h-3 bg-secondary rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-violet via-rose to-rose-glow rounded-full relative overflow-hidden" 
                    style={{ width: '70%' }}
                  >
                    <div className="absolute inset-0 animate-shimmer" />
                  </div>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="p-5 rounded-2xl bg-secondary/60 border border-border/50">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-4 h-4 text-violet" />
                    <span className="text-xs text-muted-foreground">Conversions</span>
                  </div>
                  <p className="font-display text-3xl font-normal text-foreground">12.4k</p>
                  <div className="flex items-center gap-1 mt-1 text-xs text-mint">
                    <TrendingUp className="w-3 h-3" />
                    <span>+24% vs last month</span>
                  </div>
                </div>
                <div className="p-5 rounded-2xl bg-secondary/60 border border-border/50">
                  <div className="flex items-center gap-2 mb-2">
                    <LineChart className="w-4 h-4 text-rose" />
                    <span className="text-xs text-muted-foreground">ROAS</span>
                  </div>
                  <p className="font-display text-3xl font-normal text-foreground">3.8x</p>
                  <div className="flex items-center gap-1 mt-1 text-xs text-mint">
                    <TrendingUp className="w-3 h-3" />
                    <span>Above target</span>
                  </div>
                </div>
              </div>

              {/* Creators list */}
              <div className="flex items-center justify-between">
                <div className="flex -space-x-3">
                  {['from-violet-400 to-indigo-500', 'from-rose-400 to-pink-500', 'from-amber-400 to-orange-500', 'from-sky-400 to-blue-500'].map((gradient, i) => (
                    <div
                      key={i}
                      className={`w-10 h-10 rounded-full border-3 border-white bg-gradient-to-br ${gradient} shadow-md`}
                      style={{ zIndex: 4 - i }}
                    />
                  ))}
                  <div className="w-10 h-10 rounded-full border-3 border-white bg-secondary flex items-center justify-center text-xs font-bold text-muted-foreground shadow-md">
                    +8
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">12 creators active</p>
              </div>
            </div>
          </motion.div>

          {/* Content */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="order-1 lg:order-2"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-rose-soft border border-rose/15 mb-6">
              <Building2 className="w-3.5 h-3.5 text-rose" />
              <span className="text-xs font-semibold text-rose uppercase tracking-wider">For Brands</span>
            </div>
            
            <h2 className="font-display text-display-sm sm:text-display-md font-normal mb-6 text-balance">
              Pay for <span className="italic text-rose">results</span>, not promises
            </h2>
            
            <p className="text-lg text-muted-foreground leading-relaxed mb-10 max-w-lg">
              Stop guessing which influencers will deliver. 
              See verified performance data, track every conversion, 
              and only pay when results are achieved.
            </p>

            <ul className="space-y-4">
              {benefits.map((item, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 + index * 0.08, duration: 0.5 }}
                  className="flex items-center gap-4"
                >
                  <div className="w-10 h-10 rounded-xl bg-rose-soft flex items-center justify-center flex-shrink-0">
                    <item.icon className="w-5 h-5 text-rose" />
                  </div>
                  <span className="text-foreground font-medium">{item.text}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default BrandsSection;