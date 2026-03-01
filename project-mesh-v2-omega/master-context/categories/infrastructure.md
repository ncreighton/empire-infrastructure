## Infrastructure Context
- **Services**: Dashboard (8000), Vision (8002), Grimoire (8080), VideoForge (8090), BMC (8095)
- **Startup**: VBS launchers -> PowerShell -> service binary via Task Scheduler
- **Monitoring**: Empire dashboard checks all service health every 30s
- **Deployment**: VPS at 217.216.84.245, Docker compose for remote services
