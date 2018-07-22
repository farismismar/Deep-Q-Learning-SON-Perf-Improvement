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

close all force;
clc;
clear all
clear global;
clear classes;

global enable_intelligent_SON;
global enable_efficient_handling;
%global network_data;
global Total_Time;
global q;
global live_network_alarms;

startTime = tic;

Total_Time = 50; % TTIs
live_network_alarms = true; % yes generate alarms in the network
q = 50; % UEs per cell
enable_intelligent_SON = false; 
enable_efficient_handling = true;

% Truth table
% enable_intelligent_SON, enable_efficient_handling
% F, F = Random
% F, T = FCFS
% T, F = DQN 
% T, T = unused.

global reward;
global R_min;
global R_max;
global state;
global action;
global Actions;
global alarm_register;
global cell_down_register;

% Environment entry parameters
state_size = 3;
action_size = 5;
alarm_register = [0,0,0,0,0];

R_max = 5;
R_min = -100;

EPISODE_MAX = 150;  % do not change to less.  0.01 will not be achieved.

mod = py.importlib.import_module('main'); % a pointer to main.py
py.importlib.reload(mod);

py.importlib.import_module('os');

seed = 3; % also change in main.py
rng(seed,'twister');

simulation_type = 'tri_sector_tilted_4x2';


LTE_config = LTE_load_params(simulation_type);
%% If you want to modify something taking as a base the configuration file, do it here: here an example is show that changes the inter-eNodeB distances based on the LTE_load_params_hex_grid_tilted config file.

% Some changes to the base configuration, in case you would need/want them
LTE_config.debug_level                = 1; % basic output.  
LTE_config.bandwidth                  = 10e6; % 10 MHz
LTE_config.frequency                  = 2.1e9; % 2.1 GHz
LTE_config.channel_model.type         = 'PedA';
LTE_config.use_fast_fading            = true;
LTE_config.show_network               = 3; % show plots - Everything
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
LTE_config.eNodeB_tx_power            = 10^((46-30)/10); % 46 dBm for macro
LTE_config.site_altiude               = 0;  % average terrain height 
LTE_config.site_height                = 25; % site height above terrain
LTE_config.rx_height                  = 1.5; % UE is at 1.5 meters
LTE_config.antenna_gain_pattern       = 'TS 36.942';
%LTE_config.sector_azimuths            = [30 150 270];
LTE_config.antenna.electrical_downtilt= 4;
LTE_config.max_antenna_gain           = 17; % 17 dB
LTE_config.UE.thermal_noise_density   = -174; % dBm/Hz
LTE_config.cache_network              = false;
LTE_config.antenna.antenna_type = '742212';
LTE_config.antenna.frequency = 2140;
                
% % Small cell layer
% LTE_config.add_femtocells             = true;  % femto but configured as a pico with power
% LTE_config.femtocells_config.tx_power_W = 10^((37-30)/10); % 37 dBm is 5W.
% LTE_config.femtocells_config.spatial_distribution = 'homogenous density';
% LTE_config.femtocells_config.femtocells_per_km2 = 3; %50; % 3 for case 1 and 50 for case 2
% %LTE_config.femtocells_config.macroscopic_pathloss_model = 'cost231'; % 'dual slope'

LTE_config.compact_results_file       = true;
LTE_config.delete_ff_trace_at_end     = true;
LTE_config.UE_cache                   = false;
LTE_config.UE.antenna_gain            = -1; % -1 dB.
LTE_config.UE.nRX                     = 2;  % Number of receive branches
LTE_config.UE.receiver_noise_figure   = 7; % 7dB
%LTE_config.UE_cache_file              = 'auto';
LTE_config.adaptive_RI                = 0;
LTE_config.keep_UEs_still             = false;
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

if enable_intelligent_SON == true && enable_efficient_handling == false
    py.main.set_environment(state_size,action_size)
    
    for episode_ = 1:EPISODE_MAX
        py.main.env_reset_wrapper();
        
        epsilon = py.main.agent_get_exploration_rate_wrapper();
        fprintf('Episode %d/%d.  Current epislon value is %3f:\n', episode_, EPISODE_MAX, epsilon);
    
        alarm_register = [0,0,0,0,0];
        state = zeros(1,state_size);
        for fn = fieldnames(LTE_config)'
            %LTE_config = LTE_config_reset; % reload the network settings
            %afresh. (did not work)
            LTE_config.(fn{1}) = LTE_config_reset.(fn{1}); % reload the network settings afresh.
        end

        reward = R_min;
        action = py.main.agent_begin_episode_wrapper(state);  % is always 0
        
        Actions = [0]; %zeros(1,action_size);
        output_results_file = LTE_sim_main(LTE_config);
        
        % train the agent with the experience of the episode
        if py.main.agent_memory_length_diff_wrapper() > 0
            py.main.agent_replay_wrapper();
            % Show the losses for this episode here
            losses = py.main.agent_get_losses_wrapper();
            losses = cell2mat(cell(losses));
            disp(losses)
        end
        close all
        fprintf('The list of actions for episode %d:\n', episode_);
        disp(Actions)
        filename = sprintf('actions_episode_%d.csv',episode_);
        csvwrite(filename,Actions)
    end
    
    dlmwrite('loss_opt_episode.csv', losses, '-append'); 
else
    output_results_file = LTE_sim_main(LTE_config); % This is the main line... do not re run it unless you know what you are doing.
end

%%%%%%%%%%%%%
simulation_data                   = load(output_results_file);

% Manually place sites
%simulation_data.sites(1).pos      = [0 0];  % Macro
%simulation_data.sites(2).pos      = [cos(2*pi/3) sin(2*pi/3)] * LTE_config.inter_eNodeB_distance;
%simulation_data.sites(3).pos      = [cos(240*pi/180) sin(240*pi/180)] * LTE_config.inter_eNodeB_distance;
%simulation_data.sites(4).pos      = [cos(360*pi/180) sin(360*pi/180)] * LTE_config.inter_eNodeB_distance;
close all
GUI_handles.aggregate_results_GUI = LTE_GUI_show_aggregate_results(simulation_data);
GUI_handles.positions_GUI         = LTE_GUI_show_UEs_and_cells(simulation_data,GUI_handles.aggregate_results_GUI);

% Generate the plot




% Save this data somewhere
%figure(1000)
%plot(0:Total_Time, [1;CoMPenabled], 'k')
%xlabel('TTI')
%ylabel('CoMP Decision')
%ylim([-2,2])

elapsedTime = toc(startTime);
fprintf('Simulation: total time = %1.1f sec.\n', elapsedTime);
fprintf('Simulation: quit.\n');