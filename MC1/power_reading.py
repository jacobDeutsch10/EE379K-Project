
import subprocess
import time
import telnetlib as tel


DELAY = 0.2
count = 0
SP2_tel = tel.Telnet('192.168.4.1')
total_power = 0.0
power = 0.0
max_power = -1.0

def getTelnetPower(SP2_tel, last_power):
    tel_dat = str(SP2_tel.read_very_eager())
    print("telnet reading:", tel_dat)
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2: findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power



def main():
    global power,max_power,total_power,DELAY,count
    while True:
        start = time.time()
        
        power = getTelnetPower(SP2_tel, power)
        if power > max_power:
            max_power = power
        total_power += power
        count += 1
        elapsed = time.time()-start
        time.sleep(max(0, DELAY-elapsed))

if __name__ =="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Max power: " + str(max_power))

