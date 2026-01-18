import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Shield, Zap } from "lucide-react";

const CodeVisual = () => {
  return (
    <div className="relative">
      {/* Glow effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[450px] h-[350px] bg-primary/15 rounded-full blur-[100px]" />
      
      {/* Main code block */}
      <motion.div 
        className="relative rounded-xl overflow-hidden border border-white/10 bg-[#0d1117] shadow-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-[#161b22]">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-[#ff5f56]" />
            <div className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
            <div className="w-3 h-3 rounded-full bg-[#27ca40]" />
          </div>
          <span className="text-[11px] font-mono text-white/40">auth.ts</span>
          <motion.div 
            className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-primary/15 text-primary text-[10px] font-medium"
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Zap className="w-2.5 h-2.5" />
            Scanning
          </motion.div>
        </div>
        
        {/* Code content */}
        <div className="p-4 font-mono text-[13px] leading-6">
          {/* Line 1 */}
          <div className="flex">
            <span className="w-8 text-white/20 select-none">1</span>
            <span>
              <span className="text-[#ff7b72]">async function</span>{" "}
              <span className="text-[#d2a8ff]">validateUser</span>
              <span className="text-white/60">(req) {"{"}</span>
            </span>
          </div>
          
          {/* Line 2 - vulnerable */}
          <motion.div 
            className="flex bg-primary/8 border-l-2 border-primary -mx-4 px-4"
            initial={{ opacity: 0.5 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <span className="w-8 text-white/20 select-none">2</span>
            <span>
              <span className="text-white/60 pl-4">const user = </span>
              <span className="text-primary">req.body.user</span>
              <span className="text-white/60">;</span>
              <span className="ml-3 text-[10px] text-primary/80">// ⚠ Unsanitized</span>
            </span>
          </motion.div>
          
          {/* Line 3 */}
          <div className="flex">
            <span className="w-8 text-white/20 select-none">3</span>
            <span>
              <span className="text-white/60 pl-4">const query = </span>
              <span className="text-[#a5d6ff]">`SELECT * FROM users WHERE id = ${"{user}"}`</span>
              <span className="text-white/60">;</span>
            </span>
          </div>
          
          {/* Line 4 - vulnerable */}
          <motion.div 
            className="flex bg-primary/8 border-l-2 border-primary -mx-4 px-4"
            initial={{ opacity: 0.5 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <span className="w-8 text-white/20 select-none">4</span>
            <span>
              <span className="text-white/60 pl-4">return </span>
              <span className="text-primary">db.execute(query)</span>
              <span className="text-white/60">;</span>
              <span className="ml-3 text-[10px] text-primary/80">// ⚠ SQL Injection</span>
            </span>
          </motion.div>
          
          {/* Line 5 */}
          <div className="flex">
            <span className="w-8 text-white/20 select-none">5</span>
            <span className="text-white/60">{"}"}</span>
          </div>
          
          {/* Divider */}
          <div className="my-4 border-t border-dashed border-white/10" />
          
          {/* Fixed label */}
          <motion.div 
            className="flex items-center gap-2 mb-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <CheckCircle2 className="w-3.5 h-3.5 text-success" />
            <span className="text-[11px] font-medium text-success">Sentinel Fix Applied</span>
          </motion.div>
          
          {/* Fixed code */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            <div className="flex">
              <span className="w-8 text-white/20 select-none">1</span>
              <span>
                <span className="text-[#ff7b72]">async function</span>{" "}
                <span className="text-[#d2a8ff]">validateUser</span>
                <span className="text-white/60">(req) {"{"}</span>
              </span>
            </div>
            
            <div className="flex bg-success/8 border-l-2 border-success -mx-4 px-4">
              <span className="w-8 text-white/20 select-none">2</span>
              <span>
                <span className="text-white/60 pl-4">const user = </span>
                <span className="text-success">sanitize(req.body.user)</span>
                <span className="text-white/60">;</span>
              </span>
            </div>
            
            <div className="flex bg-success/8 border-l-2 border-success -mx-4 px-4">
              <span className="w-8 text-white/20 select-none">3</span>
              <span>
                <span className="text-white/60 pl-4">return </span>
                <span className="text-success">db.query</span>
                <span className="text-white/60">('...', [user]);</span>
              </span>
            </div>
            
            <div className="flex">
              <span className="w-8 text-white/20 select-none">4</span>
              <span className="text-white/60">{"}"}</span>
            </div>
          </motion.div>
        </div>
      </motion.div>
      
      {/* Floating cards */}
      <motion.div
        initial={{ opacity: 0, x: -20, y: 10 }}
        animate={{ opacity: 1, x: 0, y: 0 }}
        transition={{ delay: 1.3, duration: 0.4 }}
        className="absolute -left-6 top-12 rounded-lg p-3 border border-white/10 bg-[#161b22] shadow-xl"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <AlertTriangle className="w-4 h-4 text-primary" />
          </div>
          <div>
            <p className="text-xs font-medium text-white">3 vulnerabilities</p>
            <p className="text-[10px] text-white/40">found in scan</p>
          </div>
        </div>
      </motion.div>
      
      <motion.div
        initial={{ opacity: 0, x: 20, y: -10 }}
        animate={{ opacity: 1, x: 0, y: 0 }}
        transition={{ delay: 1.5, duration: 0.4 }}
        className="absolute -right-6 bottom-20 rounded-lg p-3 border border-white/10 bg-[#161b22] shadow-xl"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-success/10 flex items-center justify-center">
            <Shield className="w-4 h-4 text-success" />
          </div>
          <div>
            <p className="text-xs font-medium text-white">3 tests added</p>
            <p className="text-[10px] text-white/40">regression guards</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default CodeVisual;
