using FeedBackPlatformWeb.Models;
using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;

namespace FeedBackPlatformWeb.Database
{
    public class DatabaseContext : DbContext
    {
        public DbSet<Survey> Surveys { get; set; }
        public DbSet<Category> Categories { get; set; }
        public DbSet<Question> Questions { get; set; }
        public DbSet<ClientProfile> ClientProfiles { get; set; }
        public DbSet<Response> Responses { get; set; }
    }
}