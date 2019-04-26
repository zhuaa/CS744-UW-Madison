source cluster_utils.sh

echo "Hello, "$USER". I am here to help you run Stochastic Gradient Descent"

while true
do
    echo -n "Would you like to run Single Mode(SM), Asynchronous(A) or Synchronous(S)?"
    read option
    if [ "$option" = "SM" ] || [ "$option" = "sm" ] || [ "$option" = "A" ] || [ "$option" = "a" ] || [ "$option" = "S" ] || [ "$option" = "s" ]; then
        break
    else
        echo "Please try again and enter a valid option."
    fi
done

if [ "$option" = "SM" ] || [ "$option" = "sm" ]; then
    start_cluster code_template.py single
elif [ "$option" = "A" ] || [ "$option" = "a" ]; then
    start_cluster async_training.py cluster2
elif [ "$option" = "S" ] || [ "$option" = "s" ]; then
    start_cluster sync_training.py cluster2
fi
