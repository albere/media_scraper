#!/bin/bash
while true; do
    python -m scraper "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Scraper finished successfully"
        break
    fi
    echo "Scraper exited with code $EXIT_CODE — cooling down 5 minutes..."
    sleep 300
done
