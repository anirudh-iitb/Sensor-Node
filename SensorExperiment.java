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
		String[] policies = new String[8];
		policies[0] = "greedy-policy";
		policies[1] = "stop-greedy";
		policies[2] = "mto";
		policies[3] = "stop-mto";
		policies[4] = "discounted-qlearning";
		policies[5] = "stop-discounted-qlearning";
		policies[6] = "relative-qlearning";
		policies[7] = "stop-relative-qlearning";

		double mean_data = 0.5;
		for(int j=0; j<12; j++)
		{
			for(int i=0; i<4; i++)
			{
				RLGlue.RL_init();
				String data = String.valueOf(mean_data);
				RLGlue.RL_env_message(data);		
				RLGlue.RL_agent_message(policies[2*i]);
				System.out.println("Policy is "+policies[2*i]);
				evaluateAgent();
				System.out.println("Mean data rate is "+ mean_data);
				RLGlue.RL_agent_message("print-average-cost");
				RLGlue.RL_agent_message(policies[2*i+1]);
				RLGlue.RL_cleanup();
			}
			mean_data = mean_data + 0.25;
		}
	}

	/*public void run_experiment()
	{
		RLGlue.RL_init();
		RLGlue.RL_env_message("1.5");
		RLGlue.RL_agent_message("mto");
		System.out.println("MTO");
		evaluateAgent();		
		RLGlue.RL_agent_message("print-average-cost");
		RLGlue.RL_agent_message("stop-mto");
		RLGlue.RL_cleanup();
	}*/

	public void evaluateAgent()
	{
		RLGlue.RL_start();
		for(int i=0; i<10000000; i++)
		{
			RLGlue.RL_step();
		}
		RLGlue.RL_agent_message("start-decay");
		for(int i=0; i<20000000; i++)
		{
			RLGlue.RL_step();
		}
		RLGlue.RL_agent_message("Freeze Learning");
		for(int i=0; i<10000000; i++)
		{
			RLGlue.RL_step();
		}
		RLGlue.RL_agent_message("UnFreeze Learning");
		RLGlue.RL_agent_message("stop-decay");
	}

	public static void main(String[] args)
	{
		SensorExperiment theExperiment = new SensorExperiment();
        theExperiment.run_experiment();
	}
}