using Microsoft.Owin;
using Owin;

[assembly: OwinStartupAttribute(typeof(FeedBackPlatformWeb.Startup))]
namespace FeedBackPlatformWeb
{
    public partial class Startup
    {
        public void Configuration(IAppBuilder app)
        {
            ConfigureAuth(app);
        }
    }
}
