#!/usr/bin/env bash

export _HYDRA_OLD_COMP=$(complete -p python 2> /dev/null)
echo $HYDRA_COMP_DEBUG
hydra_bash_completion()
{
    words=($COMP_LINE)
    if [ "${words[0]}" == "python" ]; then
        if (( ${#words[@]} < 2 )); then
            return
        fi
        file_path=$(pwd)/${words[1]}
        if [ ! -f "$file_path" ]; then
            return
        fi
        grep "@hydra.main" $file_path -q
        helper="${words[0]} ${words[1]}"
    else
        helper="${words[0]}"
        true
    fi

    EXECUTABLE=($(command -v $helper))
    if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
        printf "EXECUTABLE_FIRST='${EXECUTABLE[0]}'\n"
    fi
    if ! [ -x "${EXECUTABLE[0]}" ]; then
        false
    fi

    if [ $? == 0 ]; then
        choices=$( COMP_POINT=$COMP_POINT COMP_LINE=$COMP_LINE $helper -sc query=bash)
        word=${words[$COMP_CWORD]}

        if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
            printf "\n"
            printf "COMP_LINE='$COMP_LINE'\n"
            printf "COMP_POINT='$COMP_POINT'\n"
            printf "Word='$word'\n"
            printf "Output suggestions:\n"
            printf "\t%s\n" ${choices[@]}
        fi
        COMPREPLY=($( compgen -o nospace -o default -W "$choices" -- "$word" ));
    fi
}

COMP_WORDBREAKS=${COMP_WORDBREAKS//=}
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_bash_completion python
echo $COMP_WORDBREAKS