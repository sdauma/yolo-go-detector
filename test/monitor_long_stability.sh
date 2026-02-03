#!/bin/bash

# 长时间稳定性监控脚本 (bash 版本)
# 支持 Linux 和 macOS 平台

echo "=== 长时间稳定性监控脚本 ==="
echo "开始监控 Go 长时间运行稳定性..."
echo "按 Ctrl+C 停止监控"

max_rss=0
max_cpu=0
rss_values=()
cpu_values=()
start_time=$(date +%s)

trap 'echo "\n监控中断"; cleanup; exit 0' SIGINT

cleanup() {
    echo "\n=== 监控结果 ==="
    echo "最大 RSS: $(echo "scale=2; $max_rss / 1024" | bc) MB"
    echo "最大 CPU: $max_cpu%"
    
    if [ ${#rss_values[@]} -gt 0 ]; then
        # 计算稳定期 RSS（取最后 20 个值的平均值）
        stable_count=${#rss_values[@]}
        if [ $stable_count -gt 20 ]; then
            stable_count=20
        fi
        
        sum=0
        for ((i=${#rss_values[@]}-stable_count; i<${#rss_values[@]}; i++)); do
            sum=$(echo "$sum + ${rss_values[$i]}" | bc)
        done
        
        stable_rss=$(echo "scale=2; $sum / $stable_count" | bc)
        echo "稳定 RSS: $stable_rss MB"
    fi
    
    if [ ${#cpu_values[@]} -gt 0 ]; then
        # 计算稳定期 CPU（取最后 20 个值的平均值）
        stable_count=${#cpu_values[@]}
        if [ $stable_count -gt 20 ]; then
            stable_count=20
        fi
        
        sum=0
        for ((i=${#cpu_values[@]}-stable_count; i<${#cpu_values[@]}; i++)); do
            sum=$(echo "$sum + ${cpu_values[$i]}" | bc)
        done
        
        stable_cpu=$(echo "scale=2; $sum / $stable_count" | bc)
        echo "稳定 CPU: $stable_cpu%"
    fi
    
    elapsed=$(( $(date +%s) - $start_time ))
    hours=$((elapsed / 3600))
    mins=$(( (elapsed % 3600) / 60 ))
    secs=$((elapsed % 60))
    echo "总运行时间: $(printf "%02d:%02d:%02d" $hours $mins $secs)"
    echo "记录值数量: ${#rss_values[@]}"
    
    echo "=== 监控结束 ==="
}

while true; do
    # 在 Linux 上使用 ps 命令
    if [ "$(uname)" == "Linux" ]; then
        process_info=$(ps aux | grep benchmark_go_long_stability | grep -v grep)
        if [ -n "$process_info" ]; then
            current_rss=$(echo "$process_info" | awk '{print $6}')
            current_rss_mb=$(echo "scale=2; $current_rss / 1024" | bc)
            current_cpu=$(echo "$process_info" | awk '{print $3}')
            
            # 更新最大 RSS
            if (( $(echo "$current_rss > $max_rss" | bc -l) )); then
                max_rss=$current_rss
                max_rss_mb=$(echo "scale=2; $max_rss / 1024" | bc)
                echo -e "\e[31m[峰值] RSS: $max_rss_mb MB\e[0m"
            fi
            
            # 更新最大 CPU
            if (( $(echo "$current_cpu > $max_cpu" | bc -l) )); then
                max_cpu=$current_cpu
                echo -e "\e[33m[峰值] CPU: $max_cpu%\e[0m"
            fi
            
            # 记录 RSS 值
            rss_values+=($current_rss_mb)
            if [ ${#rss_values[@]} -gt 100 ]; then
                rss_values=(${rss_values[@]: -100})
            fi
            
            # 记录 CPU 值
            cpu_values+=($current_cpu)
            if [ ${#cpu_values[@]} -gt 100 ]; then
                cpu_values=(${cpu_values[@]: -100})
            fi
            
            elapsed=$(( $(date +%s) - $start_time ))
            hours=$((elapsed / 3600))
            mins=$(( (elapsed % 3600) / 60 ))
            secs=$((elapsed % 60))
            echo -e "\e[36m[实时] 运行时间: $(printf "%02d:%02d:%02d" $hours $mins $secs), RSS: $current_rss_mb MB, CPU: $current_cpu%\e[0m"
        else
            echo -e "\e[37m[状态] 未找到 Go 进程\e[0m"
        fi
    # 在 macOS 上使用 ps 命令
    elif [ "$(uname)" == "Darwin" ]; then
        process_info=$(ps aux | grep benchmark_go_long_stability | grep -v grep)
        if [ -n "$process_info" ]; then
            current_rss=$(echo "$process_info" | awk '{print $6}')
            current_rss_mb=$(echo "scale=2; $current_rss / 1024" | bc)
            current_cpu=$(echo "$process_info" | awk '{print $3}')
            
            # 更新最大 RSS
            if (( $(echo "$current_rss > $max_rss" | bc -l) )); then
                max_rss=$current_rss
                max_rss_mb=$(echo "scale=2; $max_rss / 1024" | bc)
                echo -e "\e[31m[峰值] RSS: $max_rss_mb MB\e[0m"
            fi
            
            # 更新最大 CPU
            if (( $(echo "$current_cpu > $max_cpu" | bc -l) )); then
                max_cpu=$current_cpu
                echo -e "\e[33m[峰值] CPU: $max_cpu%\e[0m"
            fi
            
            # 记录 RSS 值
            rss_values+=($current_rss_mb)
            if [ ${#rss_values[@]} -gt 100 ]; then
                rss_values=(${rss_values[@]: -100})
            fi
            
            # 记录 CPU 值
            cpu_values+=($current_cpu)
            if [ ${#cpu_values[@]} -gt 100 ]; then
                cpu_values=(${cpu_values[@]: -100})
            fi
            
            elapsed=$(( $(date +%s) - $start_time ))
            hours=$((elapsed / 3600))
            mins=$(( (elapsed % 3600) / 60 ))
            secs=$((elapsed % 60))
            echo -e "\e[36m[实时] 运行时间: $(printf "%02d:%02d:%02d" $hours $mins $secs), RSS: $current_rss_mb MB, CPU: $current_cpu%\e[0m"
        else
            echo -e "\e[37m[状态] 未找到 Go 进程\e[0m"
        fi
    else
        echo "不支持的平台: $(uname)"
        exit 1
    fi
    
    sleep 2
done