import { motion } from "framer-motion";
import { CheckCircle2, TrendingUp, Shield, Zap, CreditCard, Star, Users } from "lucide-react";

const benefits = [
  { icon: Star, text: "Build a verified performance profile" },
  { icon: CreditCard, text: "Set your own rates and terms" },
  { icon: Shield, text: "Escrow-protected payments" },
  { icon: Zap, text: "Instant withdrawals via Stripe" },
  { icon: TrendingUp, text: "Credit score shows your reliability" },
];

const CreatorsSection = () => {
  return (
    <section id="creators" className="py-24 sm:py-32 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-violet-soft/50 via-background to-background" />
      <div className="absolute inset-0 bg-gradient-mesh opacity-30" />

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 lg:gap-20 items-center">
          {/* Content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-violet-soft border border-violet/15 mb-6">
              <Users className="w-3.5 h-3.5 text-violet" />
              <span className="text-xs font-semibold text-violet uppercase tracking-wider">For Creators</span>
            </div>

            <h2 className="font-display text-display-sm sm:text-display-md font-normal mb-6 text-balance">
              Your talent,{" "}
              <span className="italic text-violet">fairly</span> rewarded
            </h2>

            <p className="text-lg text-muted-foreground leading-relaxed mb-10 max-w-lg">
              No more chasing invoices. No more undervalued work.
              Build your reputation with verified performance data
              and get paid instantly when you deliver.
            </p>

            <ul className="space-y-4 mb-10">
              {benefits.map((item, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 + index * 0.08, duration: 0.5 }}
                  className="flex items-center gap-4"
                >
                  <div className="w-10 h-10 rounded-xl bg-violet-soft flex items-center justify-center flex-shrink-0">
                    <item.icon className="w-5 h-5 text-violet" />
                  </div>
                  <span className="text-foreground font-medium">{item.text}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Visual - Creator Profile Card */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="relative"
          >
            {/* Glow effect */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-violet/10 rounded-full blur-[80px]" />

            <div className="relative glass-card-strong rounded-3xl p-8 hover-lift">
              {/* Profile Header */}
              <div className="flex items-center gap-5 mb-8">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-500 via-violet-400 to-rose-400 shadow-xl" />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-display text-2xl font-normal">Maya Rodriguez</p>
                    <CheckCircle2 className="w-5 h-5 text-violet fill-violet-soft" />
                  </div>
                  <p className="text-muted-foreground">Lifestyle • 340k followers</p>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-4 mb-8">
                {[
                  { value: "94%", label: "Delivery", color: "text-primary" },
                  { value: "4.8", label: "Rating", color: "text-gold" },
                  { value: "47", label: "Campaigns", color: "text-violet" },
                ].map((stat, i) => (
                  <div key={i} className="text-center p-4 rounded-2xl bg-secondary/60 border border-border/50">
                    <p className={`font-display text-3xl font-normal ${stat.color}`}>{stat.value}</p>
                    <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
                  </div>
                ))}
              </div>

              {/* Earnings Card */}
              <div className="flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-gray-50 to-primary/5 border border-primary/15">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Total earned</p>
                  <p className="font-display text-3xl font-normal text-foreground">$52,840</p>
                </div>
                <div className="text-right">
                  <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold mb-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Verified
                  </div>
                  <p className="text-xs text-muted-foreground">Top Creator</p>
                </div>
              </div>

              {/* Recent activity indicator */}
              <div className="mt-6 flex items-center gap-2 text-xs text-muted-foreground">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span>Active now • Last campaign 2 days ago</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default CreatorsSection;