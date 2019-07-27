% Base files are stored here: ./data_files/channel_traces/
% Once deleted, the simulation folder is small, but reproducibility is
% impacted.
% There is one more file that you need to replace for Matlab 2014a and past:
% phy_modeling/channelTraceFactory_v1.m line 561-7: 
%             if config.parallel_toolbox_installed && config.non_parallel_channel_trace && config.tx_mode~=1
%                 try
%                     parpool open;
%                 catch 
%                     fprintf('Failed to open matlabpool. Maybe already open?\n');
%                 end
%             end
% If you see an issue with gfortran, then rename
% /Applications/MATLAB_R2016b.app/sys/os/maci64/libgfortran.3.dylib to 
% libgfortran.3.dylib.old to let Matlab search for the gcc gfortran (which
% you need to install from gcc).
% The core is 13 files that are different from the LTE-A sim.

% Note if trying tensorflow and failing with this error:
% Error using pywrap_tensorflow><module> (line 74)
% Then run MATLAB from Terminal.
% /Applications/MATLAB_R2018a.app/bin/matlab

% First time run the random algorithm.  Do not start by running the
% intelligent SON.
% Change folder do not add to path.

close all force;
clc;
clear all
clear global;
clear classes;

global enable_intelligent_SON;
global enable_fcfs_handling;
global Total_Time;
global q;
global live_network_alarms;

startTime = tic;

Total_Time = 20; % TTIs
live_network_alarms = true; % yes generate alarms in the network.  No will be upper bound.

q = 20; % UEs per cell
enable_intelligent_SON = true;
enable_fcfs_handling = false;

% Truth table
% enable_intelligent_SON, enable_fcfs_handling
% F, F = Random  
% F, T = FCFS  /  FIFO
% T, F = DQN 
% T, T = unused will yield error.

global total_reward;
global R_min;
global R_max;
global state;
global action;
global Actions
global Rewards
global Alarms
global alarm_register;
global state_size;
global action_size;
global Q_value
global losses
    
global EPISODE_MAX;

% Environment entry parameters
state_size = 3;
action_size = 6;
alarm_register = zeros(1,action_size);

R_max = 5;
R_min = -2;

EPISODE_MAX = 1000;  % do not change to less.  0.01 will not be achieved.

mod = py.importlib.import_module('main'); % a pointer to main.py
py.importlib.reload(mod);

py.importlib.import_module('os');

seed = 0; % also change in main.py
rng(seed,'twister');

simulation_type = 'tri_sector_tilted_4x2';

LTE_config = LTE_load_params(simulation_type);
%% If you want to modify something taking as a base the configuration file, do it here: here an example is show that changes the inter-eNodeB distances based on the LTE_load_params_hex_grid_tilted config file.

% Some changes to the base configuration, in case you would need/want them
LTE_config.debug_level                = 1; % basic output.  
LTE_config.bandwidth                  = 10e6; % 10 MHz
LTE_config.frequency                  = 2.1e9; % 2.1 GHz
LTE_config.channel_model.type         = 'PedA';
LTE_config.use_fast_fading            = false;
LTE_config.show_network               = 0; % do not show plots - Everything
LTE_config.feedback_channel_delay     = 1;  % see if this helps in computing the SINR.
LTE_config.nTX                        = 2;
LTE_config.nRX                        = 2; 
LTE_config.tx_mode                    = 4;  % 4 = CLSM
LTE_config.seedRandStream             = true; % Allow reproducibility
LTE_config.RandStreamSeed             = seed;  
LTE_config.scheduler                  = 'prop fair Sun'; 
LTE_config.network_source             = 'generated'; % hexagonals
LTE_config.network_geometry           = 'regular_hexagonal_grid';
LTE_config.nr_eNodeB_rings            = 1; 
LTE_config.inter_eNodeB_distance      = 200; % 200m apart.
LTE_config.antenna_azimuth_offsett    = 30;  % Changes the reference of the azimuth at 0 degrees.
LTE_config.macroscopic_pathloss_model = 'cost231'; % Good for HN and 2100 MHz simulations.
LTE_config.macroscopic_pathloss_model_settings.environment = 'urban_macro'; %for cost231.
LTE_config.shadow_fading_type         = 'claussen';
LTE_config.shadow_fading_mean         = 0;
LTE_config.shadow_fading_sd           = 8; % 8 dB
LTE_config.eNodeB_tx_power            = 40; % 40W for macro
LTE_config.site_altiude               = 0;  % average terrain height 
LTE_config.site_height                = 25; % site height above terrain
LTE_config.rx_height                  = 1.5; % UE is at 1.5 meters
LTE_config.antenna_gain_pattern       = 'TS 36.942'; 
LTE_config.antenna.electrical_downtilt= 4;
LTE_config.max_antenna_gain           = 17; % 17 dB
LTE_config.UE.thermal_noise_density   = -174; % dBm/Hz
LTE_config.cache_network              = true;
LTE_config.antenna.antenna_type = '742212';
LTE_config.antenna.frequency = 2140;
                
LTE_config.compact_results_file       = true;
LTE_config.delete_ff_trace_at_end     = true;
LTE_config.UE_cache                   = false;
LTE_config.UE.antenna_gain            = -1; % -1 dB.
LTE_config.UE.nRX                     = 2;  % Number of receive branches
LTE_config.UE.receiver_noise_figure   = 7; % 7dB
%LTE_config.UE_cache_file              = 'auto';
LTE_config.adaptive_RI                = 0;
LTE_config.keep_UEs_still             = true;
LTE_config.UE_per_eNodeB              = q; % Number of UEs per cell.
LTE_config.UE_speed                   = 3/3.6; % Speed at which the UEs move. In meters/second: 5 Km/h = 1.38 m/s
LTE_config.map_resolution             = 5;  % 1 is the highest resolution.
LTE_config.pregenerated_ff_file       = 'auto';
LTE_config.trace_version              = 'v1';    % 'v1' for pregenerated precoding. 'v2' for run-time-applied precoding (generates error)

LTE_config.simulation_time_tti        = Total_Time;
%%%%%%%%%%%%%

% Create a copy of LTE_config to be reset every episode.
for fn = fieldnames(LTE_config)'
    LTE_config_reset.(fn{1}) = LTE_config.(fn{1});
end

%%%%%%%%%%%%%
if enable_intelligent_SON == true
    py.main.set_environment(state_size,action_size)
    losses = [];
    Q_value = [];
    best_episode = 0;
    best_reward = -1;
end

for episode_ = 1:EPISODE_MAX
    if enable_intelligent_SON == true
        py.main.env_reset_wrapper();

        epsilon = py.main.agent_get_exploration_rate_wrapper();
        fprintf('Episode %d/%d.  Current epislon value is %3f:\n', episode_, EPISODE_MAX, epsilon);

        % Start afresh...
        % This is env.reset()
        alarm_register = zeros(1,action_size);
        state = zeros(1,state_size);
        total_reward = 0;
        Alarms = [];
        losses = [];
        Q_value = [];

        action = -1;
        Actions = [action]; %zeros(1,action_size);
        Rewards = [0];

        py.main.agent_begin_episode_wrapper(state);  % is always 0
     end  % if SON 

    output_results_file = LTE_sim_main(LTE_config); % This interacts with the  env

    if enable_intelligent_SON == true    
        successful = (sum(alarm_register) == 0);
        if successful
            total_reward = Rewards(end);
            total_reward = total_reward + R_max;
        end
        Rewards = [Rewards(1:end-1), total_reward];

        if (total_reward > best_reward)
            best_episode = episode_;
            best_reward = total_reward;
        end
        %close all;
        fprintf('The list of actions/rewards for episode %d:\n', episode_);
        fprintf('Actions: ')
        fprintf('%d,', Actions)
        fprintf('\n');
        fprintf('Rewards: ')
        fprintf('%d,', Rewards)
        fprintf('\n');
        fprintf('Count of unresolved alarms: ')
        fprintf('%d,', Alarms)

        loss_z = mean(losses);
        q_z = mean(Q_value);
        fprintf('\n\n');
        fprintf('The loss in this episode was %3f.\n', loss_z);
        fprintf('The Q-value in this episode was %3f.\n', q_z);

        fid = fopen(sprintf('actions_episode_%d.csv',episode_), 'wt');
        if fid ~= -1
            fprintf(fid, 'Episode:,%d', episode_);
            fprintf(fid, '\nActions:,');
            fprintf(fid, '%d,', Actions);
            fprintf(fid, '\nRewards:,');
            fprintf(fid, '%d,', Rewards);
            fprintf(fid, '\nLosses:,');
            fprintf(fid, '%5f,', losses);
            fprintf(fid, '\nQ-value:,');
            fprintf(fid, '%5f,', Q_value);
            fclose(fid);
        end

        if (sum(Alarms) == 0) || isnan(loss_z)
            break
        end

        fid = fopen('best_episode.csv', 'wt');
        if fid ~= -1
            fprintf(fid, 'Best episode:, %d\n', best_episode);
            fprintf(fid, 'Best reward:, %6f\n', total_reward);
            fclose(fid);
        end
    end % enable_SON
end % for episode

%%%%%%
simulation_data                   = load(output_results_file);
% TODO: Try to find where the throughput and the SE CDF is and port it here.

close all
GUI_handles.aggregate_results_GUI = LTE_GUI_show_aggregate_results(simulation_data);
GUI_handles.positions_GUI         = LTE_GUI_show_UEs_and_cells(simulation_data,GUI_handles.aggregate_results_GUI);

elapsedTime = toc(startTime);
fprintf('Simulation: total time = %1.1f sec.\n', elapsedTime);
fprintf('Simulation: quit.\n');