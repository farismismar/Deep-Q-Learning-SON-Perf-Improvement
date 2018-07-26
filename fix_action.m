function cleared = fix_action(eNodeBs, serv_cell, neigh_cell, action)

    global alarm_register
    global cell_down_register
    
    fprintf('DEBUG: Action chosen is %d.  ', action)
   
    cleared = false;
    if (action <= 1)
         % do nothing.
       % fprintf('(no action taken).\n')
    elseif (action == 2) && (alarm_register(2) == 1)
            if (eNodeBs(neigh_cell).always_on == 0) %(eNodeBs(neigh_cell).max_power == -inf) % %||  
                eNodeBs(neigh_cell).max_power = 10^((46-30)/10); % 46 dBm in Watts -- cell is back on again
                % reschedule users
                eNodeBs(neigh_cell).always_on = 1;
                fprintf('CLEARED: Neighbor cell %d is up again.\n', neigh_cell)
                cell_down_register(neigh_cell) = 0;                
                if (sum(cell_down_register) == 0)
                    alarm_register(2) = 0; 
                    cleared = true;
                end
            else
                %fprintf('(network status is unchanged)\n')
            end
    elseif (action == 3) && (alarm_register(3) == 1)
            if (eNodeBs(serv_cell).total_nTX == 1)
                eNodeBs(serv_cell).total_nTX = 2; % the cell is configured for transmit div
                fprintf('CLEARED: Serving cell transmit diversity enabled.\n')
                cleared = true;
                alarm_register(3) = 0;
            else
                %fprintf('(network status is unchanged)\n')
            end
    elseif (action == 4) && (alarm_register(4) == 1)
            if (eNodeBs(serv_cell).antenna.max_antenna_gain ~= 17)
                eNodeBs(serv_cell).antenna.max_antenna_gain = 17; 
                fprintf('CLEARED: Serving cell losses recovered.\n')
                cleared = true;
                alarm_register(4) = 0;
            else
                %fprintf('(network status is unchanged)\n')
            end                            
    elseif (action == 5) && (alarm_register(5) == 1)
            if (eNodeBs(serv_cell).azimuth ~= 150)
                eNodeBs(serv_cell).azimuth = 150;  % changed azimuth back to default
                fprintf('CLEARED: Serving cell azimuth change recovered from wind back to 150.\n')
                cleared = true;
                alarm_register(5) = 0;
            else
               % fprintf('(network status is unchanged)\n')
            end
    end
end