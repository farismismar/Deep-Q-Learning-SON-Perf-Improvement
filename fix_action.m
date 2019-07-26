function cleared = fix_action(eNodeBs, serv_cell, action)

    global alarm_register
       
    %fprintf('DEBUG: Action chosen is %d.', action)
   
    cleared = false;
    if (action == 0)
         % do nothing.
       % fprintf('(no action taken).\n')
    elseif (action == 1) && (alarm_register(1) > 0)
         if (eNodeBs(serv_cell).max_power ~= 40)
            eNodeBs(serv_cell).max_power = 40; % removed 3 dB loss of sig
            fprintf('CLEARED: Cell %d alarm loss of signal normal.\n', serv_cell)
            cleared = true;
            alarm_register(1) = min(0, alarm_register(1) - 1);
         end
    elseif (action == 2) && (alarm_register(2) > 0)
        % find the correct azimuth
        azimuths = [30, 150, 270];
        corr_azimuth = azimuths(serv_cell - 12); % since we are doing cells 13, 14, 15.
        if (eNodeBs(serv_cell).azimuth ~= corr_azimuth)
            eNodeBs(serv_cell).azimuth = corr_azimuth;
            fprintf('CLEARED: Cell %d azimuth has changed back to %d.\n', serv_cell, corr_azimuth)
            cleared = true;
            alarm_register(2) = min(0, alarm_register(2) - 1);
        end
    elseif (action == 3) && (alarm_register(3) > 0)
           if (eNodeBs(serv_cell).electrical_downtilt == 8)
                eNodeBs(serv_cell).electrical_downtilt = 4;
                fprintf('CLEARED: Cell %d tilt has changed back to 4 degrees.\n', serv_cell)
                cleared = true;
                alarm_register(3) = min(0, alarm_register(3) - 1);  
           end
    elseif (action == 4) && (alarm_register(4) > 0)
            if (eNodeBs(serv_cell).max_power ~= 37)
                eNodeBs(serv_cell).max_power = 37; % recovered
                fprintf('CLEARED: Cell %d amplifier overpower alarm 3 dB.\n', serv_cell)
                cleared = true;
                alarm_register(4) = min(0, alarm_register(4) - 1);
            end
    elseif (action == 5) && (alarm_register(5) > 0)
            if (eNodeBs(serv_cell).total_nTX == 1)
                eNodeBs(serv_cell).total_nTX = 2; % the cell is not configured for transmit div
                fprintf('CLEARED: Cell %d transmit diversity enabled.\n', serv_cell)
                cleared = true;
                alarm_register(5) = min(0, alarm_register(5) - 1);
            end
    end