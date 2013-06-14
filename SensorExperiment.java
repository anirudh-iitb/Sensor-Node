import java.lang.Integer;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;
import org.rlcommunity.rlglue.codec.RLGlue;

public class SensorExperiment
{
	public void run_experiment()
	{
		RLGlue.RL_init();
		evaluateAgent();
		RLGlue.RL_agent_message("print-policy");
		//RLGlue.RL_agent_message("print-value-function");
		RLGlue.RL_agent_message("print-average-cost");
		RLGlue.RL_agent_message("print-maximum-visited-state");
		RLGlue.RL_cleanup();
	}

	public void evaluateAgent()
	{
		RLGlue.RL_start();
		for(int i=0; i<30000000; i++)
		{
			RLGlue.RL_step();
		}
		RLGlue.RL_agent_message("Freeze Learning");
		for(int i=0; i<1000000; i++)
		{
			RLGlue.RL_step();
		}
		RLGlue.RL_agent_message("UnFreeze Learning");
	}

	public static void main(String[] args)
	{
		SensorExperiment theExperiment = new SensorExperiment();
        theExperiment.run_experiment();
	}
}