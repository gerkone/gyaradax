#!/bin/bash

# Just execute this script without arguments.
# ./init_performance_testcases.sh


cd $GKW_HOME/tests/performance

testcases_list=(    nonspec_nl kin_nl_bpar collisions_please_dont_break_me kitchen_sink)
ntime_factor_list=( 80         8           35                              6           )

for i in $(seq 0 $(( ${#testcases_list[@]} - 1 )))
do
    testcase=${testcases_list[$i]}
    echo "----------------- $testcase ----------------------"
    testcase_folder=$GKW_HOME/tests/standard/$testcase
    if [ -d $testcase_folder ]
    then
        mkdir -p $testcase/{1,reference}
        
        cp $testcase_folder/* $testcase/ &> /dev/null

        ntime=$( grep -i ntime $testcase/input.dat | sed -e 's|^.*=\([^!]*\)\s*\(!.*\)\?|\1|' -e 's|,\s*$||' )
        naverage=$( grep -i naverage $testcase/input.dat | sed -e 's|^.*=\([^!]*\)\s*\(!.*\)\?|\1|' -e 's|,\s*$||' )
        total_timesteps=$(( $ntime * $naverage ))

        new_naverage=$naverage
        #new_naverage=2

        # adjust a few parameters:

        # call the diagnostics often, too
gkw_switch_testcase_parameters $testcase > /dev/null <<EOF
naverage
CONTROL
$new_naverage
y
EOF

        # make the runs last long enough to get a reasonable average
        ntime_factor=${ntime_factor_list[$i]}
gkw_switch_testcase_parameters $testcase > /dev/null <<EOF
ntime
CONTROL
$(( ($total_timesteps * $ntime_factor) / $new_naverage ))
y
EOF

        # do not stop too quickly
gkw_switch_testcase_parameters $testcase > /dev/null <<EOF
gamatol
CONTROL
0
y
EOF

        # enable builtin performance timing
        gkw_switch_testcase_parameters $testcase > /dev/null <<EOF
iperform_set
CONTROL
-1
y
EOF

        # disable disk IO, at least for diagnostics
        gkw_switch_testcase_parameters $testcase > /dev/null <<EOF
io_format
CONTROL
'none'
y
EOF

        # print the differences to the screen
        echo diff $testcase_folder/input.dat $PWD/$testcase/input.dat
        diff $testcase_folder/input.dat $PWD/$testcase/input.dat

    else
        echo "Testcase $testcase_folder does not exist"
    fi

done

function read-yesno(){
    varname="$1"

    while true; do
	read tmp
	if [ "$tmp" == "y" ]; then
	    eval $varname="y"
	    return
	else
	    eval $varname="n"
	    return
	fi
    done
}

echo "Do you want to run the tests now and copy the performance data to the respective reference folder? [y/n]"
read-yesno answer

if [ "$answer" == "y" ]
then

    for testcase in ${testcases_list[@]}
    do
        gkw_run_tests $testcase
        cp -v $testcase/1/{perform.dat,resource_usage} $testcase/reference
    done

    echo "You may now run"
    echo "gkw_run_tests -s --ignore='^\$' --diff-command=\"numdiff -V -r 0.01 -a 1.0 --strict\" \$GKW_HOME/tests/performance"
fi
