from SemanticCascadeProcessing import SemanticCascadeProcessor, SCPConfig
from scp_diagnostics import SCPDiagnostics
import time

# Initialize SCP with diagnostics
scp = SemanticCascadeProcessor(SCPConfig(debug_mode=True))
diagnostics = SCPDiagnostics(scp)

# Process with monitoring and save report
user_input = "How might humans use mycelium to evolve the ability to survive in the void of space without the use of spacesuits or spaceships? Provide your scientific breakdown."
diagnostics.monitor_interaction(user_input)
report = diagnostics.generate_detailed_report()

# Export metrics as JSON
metrics_json = diagnostics.export_layer_metrics()
timestamp = time.strftime("%Y%m%d-%H%M%S")
with open(f'scp_metrics_{timestamp}.json', 'w') as f:
    f.write(metrics_json)

# Analyze corpus state
corpus_analysis = diagnostics.analyze_corpus_state()
print(corpus_analysis)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Rest of your code...
